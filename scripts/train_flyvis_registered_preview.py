"""One declared epoch through whole TRAIN histories, adapting only author-free parameters.

The existing auxiliary readout is frozen and is not a motor policy. This experiment
does not qualify physiological function, live control, or unseen-track performance.
"""

import argparse
import inspect
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.turn_residual import current_turn_prior
from scripts.build_retinotopic_preview_dataset import read
from scripts.probe_flyvis_registered_gradient import detached_state_exact, registered_features
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_column_preview_comparison import metrics
from scripts.train_visual_body_preview import body_features, direct_body_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (plan['epochs'] != 1 or plan['chunk_intervals'] != 2
            or plan['learning_rate'] != 1e-5 or plan['readout_trainable'] is not False):
        raise ValueError('Fixed one-epoch, two-interval, frozen-readout experiment required')
    for path, digest in plan['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Declared source/input changed: '+path)
    data, model_dir = Path(plan['data']), Path(plan['model_dir'])
    source = read(data/'dataset.json')
    sensor = read(Path(plan['sensor_replay'])/'result.json')
    episodes = []
    for e in source['episodes']:
        if e['role'] not in ('train', 'development'):
            raise ValueError('No evaluation access or role conversion')
        ref = next(r for r in sensor['episodes'] if r['id'] == e['id'])
        response = Path(ref['output'])/'responses.npz'
        for path, digest in ((response, ref['responses_sha256']),
                             (e['features_file'], e['features_sha256']),
                             (e['targets_file'], e['targets_sha256'])):
            if sha256(path) != digest:
                raise ValueError('Original whole history changed')
        with load_array(response, allow_pickle=False) as a:
            clock = a['decision_monotonic_seconds'].copy()
            retina = a['retina_intensity'].copy()
        with load_array(e['targets_file'], allow_pickle=False) as a:
            np.testing.assert_array_equal(clock, a['observation_time'])
            target = a['preview'].reshape(-1, 6).copy()
            valid = np.repeat(a['preview_valid'], 2, axis=1)
        with load_array(e['features_file'], allow_pickle=False) as a:
            np.testing.assert_array_equal(clock, a['observation_time'])
            baseline_features = a['registered'].copy()
        body, mask, _ = direct_body_reference(e, clock, sha256(plan['body_contract']))
        if len(clock) != e['observations'] or len(retina) != len(clock):
            raise ValueError('All original observations and held-input intervals required')
        episodes.append(dict(e, clock=clock, retina=retina, y=target, valid=valid,
            body=body, body_valid=mask, vision=Path(ref['output']), baseline=baseline_features))
    train = [e for e in episodes if e['role'] == 'train']
    development = [e for e in episodes if e['role'] == 'development']
    if [e['id'] for e in train] != plan['train_ids'] or len(development) != 1:
        raise ValueError('Declared TRAIN order and one known DEVELOPMENT required')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), development_used_to_fit=False,
        original_sources=source['episodes']))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'executed-gradient-helpers.py').write_bytes(
        Path('scripts/probe_flyvis_registered_gradient.py').read_bytes())
    result = dict(status='failed', author_optimizer_updates=0, readout_optimizer_updates=0,
        actor_deployed=False, motor_transfer_qualified=False, goal_achieved=False,
        physiological_function_verified=False, independent_final_validation=False)
    updates, records = 0, []
    try:
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required')
        os.environ['FLYVIS_ROOT_DIR'] = str(Path(plan['cache_root']).resolve())
        os.environ['MPLBACKEND'] = 'Agg'
        install_datamate_cache_compat(plan['cache_root'])
        import flyvis

        torch.manual_seed(plan['seed'])
        torch.set_default_device('cuda')
        config = yaml.safe_load((model_dir/'_meta.yaml').read_text(encoding='utf8'))
        network = flyvis.Network(**config['config']['network']).to('cuda')
        flags = {name: p.requires_grad for name, p in network.named_parameters()}
        weights = torch.load(model_dir/'best_chkpt', map_location='cpu',
                             weights_only=True)['network']
        network.load_state_dict(weights, strict=True)
        implementation = Path(inspect.getsourcefile(type(network)))
        write_json(run/'author-implementation.json', dict(path=str(implementation),
            sha256=sha256(implementation), original_trainable_flags=flags))
        saved = torch.load(plan['readout'], map_location='cuda', weights_only=False)
        readout = RegisteredPreview().to('cuda').eval().requires_grad_(False)
        readout.load_state_dict(saved['model'])
        optimizer = torch.optim.Adam([p for name, p in network.named_parameters() if flags[name]],
                                     lr=plan['learning_rate'])
        originals = {name: p.detach().clone() for name, p in network.named_parameters()
                     if flags[name]}
        parameter_scales = {name: p.abs().mean().clamp_min(.01) for name, p in originals.items()}
        cal = Path(read(data/'manifest.json')['config']['plan']['reference_data'])
        with load_array(cal/'channel-calibration.npz', allow_pickle=False) as a:
            mean, scale = a['processed_actual_mean'].copy(), a['processed_actual_scale'].copy()
        feature_mean = torch.tensor(saved['feature_mean'], dtype=torch.float32)
        feature_scale = torch.tensor(saved['feature_scale'], dtype=torch.float32)
        target_mean = torch.tensor(saved['target_mean'], dtype=torch.float32)
        target_scale = torch.tensor(saved['target_scale'], dtype=torch.float32)
        counts = np.concatenate([e['valid'] for e in train]).sum(0)
        weight = torch.tensor(1./(6*counts), dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats()

        def setup(e):
            network.eval().requires_grad_(False)
            with torch.no_grad():
                state = network.steady_state(1., .005, batch_size=4, value=.5)
            with load_array(e['vision']/'retina-registration.npz', allow_pickle=False) as a:
                indices = np.concatenate([a['additional_indices'], a['motion_indices']], axis=0)
                rays, origins = a['body_directions'].copy(), a['body_origins_m'].copy()
            b = body_features(e['body'], e['body_valid'], saved['body_mean'], saved['body_scale'],
                              saved['plan']['body_route_indices'])
            return state, torch.tensor(indices, dtype=torch.long), rays, origins, torch.tensor(b)

        def features(e, index, state, indices, rays, origins):
            selected = state.nodes.activity[:, indices]
            return registered_features(selected, rays, origins, e['body'][index:index+1],
                e['body_valid'][index:index+1], mean, scale)

        def advance(e, index, state, params):
            network.stimulus.zero(4, 1)
            network.stimulus.add_input(torch.tensor(e['retina'][index])[:, None, None, :])
            stimulus = network.stimulus()[:, 0].clone()
            intervals = integration_intervals(float(e['clock'][index+1]-e['clock'][index]))
            for dt in intervals:
                state = network._next_state(params, state, stimulus, dt)
            return state, len(intervals)

        def predict(x, b):
            standardized = ((x-feature_mean)/feature_scale).clamp(-5, 5)
            return readout(standardized, b)

        for e in train:
            wall = time.perf_counter()
            state, indices, rays, origins, b = setup(e)
            # Initial observation is retained but its gray initial state is detached.
            # It cannot supply an internal-network gradient in truncated training.
            with torch.no_grad():
                initial = predict(features(e, 0, state, indices, rays, origins), b[:1])
                initial_loss = ((initial-torch.tensor(
                    (e['y'][:1]-saved['target_mean'])/saved['target_scale'],
                    dtype=torch.float32)).square()*torch.tensor(e['valid'][:1])*weight).sum()
            for name, p in network.named_parameters():
                p.requires_grad_(flags[name])
            network.clamp()
            losses, ticks, gradient_rows = [], 0, np.zeros(6, int)
            for start in range(0, len(e['clock'])-1, plan['chunk_intervals']):
                optimizer.zero_grad(set_to_none=True)
                params = network._param_api()
                end = min(start+plan['chunk_intervals'], len(e['clock'])-1)
                loss = torch.zeros((), dtype=torch.float32)
                for i in range(start, end):
                    state, count = advance(e, i, state, params)
                    ticks += count
                    x = features(e, i+1, state, indices, rays, origins)
                    output = predict(x, b[i+1:i+2])
                    target = torch.tensor(e['y'][i+1:i+2], dtype=torch.float32)
                    valid = torch.tensor(e['valid'][i+1:i+2])
                    squared = (output-(target-target_mean)/target_scale).square()
                    loss = loss+(squared*valid*weight).sum()
                    gradient_rows += e['valid'][i+1].astype(int)
                regularizer = torch.stack([
                    ((p-originals[name])/parameter_scales[name]).square().mean()
                    for name, p in network.named_parameters() if flags[name]]).mean()
                objective = loss+plan['relative_parameter_regularization']*regularizer
                if not torch.isfinite(objective):
                    raise ValueError('Nonfinite author learning objective')
                objective.backward()
                for name, p in network.named_parameters():
                    if not flags[name] and p.grad is not None:
                        raise ValueError('Fixed author parameter received a gradient')
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise ValueError('Nonfinite author learning gradient')
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for name, p in network.named_parameters() if flags[name]],
                    plan['gradient_norm_limit'], error_if_nonfinite=True)
                state, leaves = detached_state_exact(state)
                optimizer.step()
                network.clamp()
                updates += 1
                losses.append(float(loss.detach()))
                if updates % 20 == 0:
                    write_json(run/'progress.json', dict(episode=e['id'], role='train',
                        decision=end, author_optimizer_updates=updates, loss=losses[-1],
                        gradient_norm_before_limit=float(norm), detached_state_tensors=leaves))
            records.append(dict(episode=e['id'], role='train', observations=len(e['clock']),
                covered_original_intervals=len(e['clock'])-1, internal_ticks=ticks,
                gradient_target_counts=gradient_rows.tolist(),
                initial_observation_loss=float(initial_loss),
                detached_initial_target_counts=e['valid'][0].astype(int).tolist(),
                online_preupdate_losses=losses, wall_seconds=time.perf_counter()-wall,
                recurrent_resets_inside_episode=0))
            write_json(run/'completed-train.json', dict(records=records, updates=updates))
        network.eval().requires_grad_(False)
        changed = [k for k, value in network.state_dict().items()
                   if k in weights and not torch.equal(value.detach().cpu(), weights[k].cpu())]
        if not changed or any(not flags.get(k, False) for k in changed):
            raise ValueError('No author adaptation or originally fixed value changed')
        if any(p.grad is not None for p in readout.parameters()):
            raise ValueError('Frozen readout received a gradient')
        torch.save(dict(network=network.state_dict(), plan=plan, original_trainable_flags=flags,
            author_optimizer_updates=updates, readout_trainable=False, actor_policy=False),
            run/'adapted-author.pt')
        reports = []
        for e in development:
            state, indices, rays, origins, b = setup(e)
            whole = []
            with torch.no_grad():
                params = network._param_api()
                for i in range(len(e['clock'])):
                    if i:
                        state, _ = advance(e, i-1, state, params)
                    whole.append(features(e, i, state, indices, rays, origins).cpu().numpy()[0])
                    if i % 100 == 0:
                        write_json(run/'progress.json', dict(episode=e['id'], role='development',
                            decision=i, author_optimizer_updates=updates, fitting=False))
            whole = np.asarray(whole)
            predictions = {}
            feature_mean_control = ((whole-saved['feature_mean'])/
                                    saved['feature_scale']).clip(-5, 5)
            feature_mean_control[:, :896] = 0
            raw_mean = whole.copy()
            raw_mean[:, :896] = 0
            with torch.no_grad():
                for intervention, x in dict(adapted=whole, original=e['baseline'],
                                             retinal_neural_mean=raw_mean).items():
                    predictions[intervention] = predict(torch.tensor(x), b).cpu().numpy()
                predictions['neural_feature_mean'] = readout(
                    torch.tensor(feature_mean_control, dtype=torch.float32), b).cpu().numpy()
            for k in predictions:
                predictions[k] = predictions[k]*saved['target_scale']+saved['target_mean']
            predictions['prior_only'], available = current_turn_prior(e['body'], e['body_valid'])
            for key, inferred in predictions.items():
                errors = e['y']-inferred
                score = float(np.mean([np.mean((errors[e['valid'][:, h], h]/
                    saved['target_scale'][h])**2) for h in range(6)]))
                reports.append(dict(episode=e['id'], role='development', intervention=key,
                    standardized_mse=score, prior_available_rows=int(available.sum()),
                    **metrics(e['y'], e['valid'], e['clock'], inferred, saved['target_mean'])))
            np.savez_compressed(run/(e['id']+'-predictions.npz'), **predictions)
            np.savez_compressed(run/(e['id']+'-adapted-features.npz'),
                                features=whole, observation_time=e['clock'])
        result.update(status='completed_author_registered_preview_adaptation',
            author_optimizer_updates=updates, records=records, reports=reports,
            author_parameters_changed=changed, original_fixed_parameters_unchanged=True,
            peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated(),
            recurrent_resets_inside_episode=0, development_used_to_fit=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error),
                      author_optimizer_updates=updates, completed_train=records)
        raise
    finally:
        write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
