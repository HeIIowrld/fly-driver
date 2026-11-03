"""Matched future-path residual diagnostics; not an actor or a fly motor policy."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.turn_residual import current_turn_prior, residual_weights
from scripts.build_retinotopic_preview_dataset import read
from scripts.diagnose_registered_anticipation import anticipation_groups
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_column_preview_comparison import metrics
from scripts.train_visual_body_preview import body_features, body_moments, direct_body_reference

CONDITIONS = ['body_uniform', 'registered_uniform', 'body_balanced', 'registered_balanced']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (plan['conditions'] != CONDITIONS or plan['epochs'] != 200
            or plan['candidate'] != 'registered_balanced'):
        raise ValueError('Four fixed-last200 conditions and predeclared candidate required')
    for file, digest in plan['source_sha256'].items():
        if sha256(file) != digest:
            raise ValueError('Predeclared source changed')
    dataset = Path(plan['data'])/'dataset.json'
    if sha256(dataset) != plan['dataset_sha256']:
        raise ValueError('Whole declared dataset changed')
    source = read(dataset)
    if (source['schema'] != 'flybrain.body_registered_preview.v1'
            or sorted(e['role'] for e in source['episodes']) != [
                'development', 'train', 'train', 'train']):
        raise ValueError('Original three TRAIN and one known DEVELOPMENT histories required')
    episodes = []
    for entry in source['episodes']:
        for key in ('features', 'targets'):
            if sha256(entry[key+'_file']) != entry[key+'_sha256']:
                raise ValueError('Whole source features or targets changed')
        with load_array(entry['features_file'], allow_pickle=False) as x, load_array(
                entry['targets_file'], allow_pickle=False) as y:
            clock = x['observation_time'].copy()
            np.testing.assert_array_equal(clock, y['observation_time'])
            features = {k: x[k].copy() for k in x.files if k != 'observation_time'}
            target = y['preview'].reshape(-1, 6).copy()
            valid = np.repeat(y['preview_valid'], 2, axis=1)
        body, mask, _ = direct_body_reference(entry, clock, sha256(plan['body_contract']))
        prior, available = current_turn_prior(body, mask)
        episodes.append(dict(entry, x=features, y=target, valid=valid, clock=clock,
                             body=body, body_valid=mask, prior=prior, available=available))
    train = [e for e in episodes if e['role'] == 'train']
    y, valid, prior = (np.concatenate([e[k] for e in train]) for k in ('y', 'valid', 'prior'))
    ym = np.array([y[valid[:, h], h].mean() for h in range(6)])
    ys = np.maximum([y[valid[:, h], h].std() for h in range(6)], plan['target_scale_floor'])
    residual = y-prior
    rm = np.array([residual[valid[:, h], h].mean() for h in range(6)])
    bm, bs, body_counts = body_moments(episodes)
    x = np.concatenate([e['x']['registered'] for e in train])
    xm, xs = x.mean(0, dtype=float), x.astype(float).std(0).clip(1e-5)
    xm[896:], xs[896:] = 0., 1.
    for e in episodes:
        e['b'] = body_features(e['body'], e['body_valid'], bm, bs, [0, 1, 4, 16])
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        dataset_sha256=sha256(dataset), program_sha256=sha256(__file__),
        device=torch.cuda.get_device_name(0), original_sources=source['episodes'],
        body_train_valid_counts=body_counts.tolist(), development_used_to_fit=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'executed-turn-residual.py').write_bytes(
        Path('src/flybrain_drive/turn_residual.py').read_bytes())
    tx = torch.tensor(((x-xm)/xs).clip(-5, 5), device='cuda', dtype=torch.float32)
    tb = torch.tensor(np.concatenate([e['b'] for e in train]), device='cuda')
    ty = torch.tensor((residual-rm)/ys, device='cuda', dtype=torch.float32)
    reports, records = [], []
    for condition in CONDITIONS:
        body_only = condition.startswith('body_')
        weights, counts = residual_weights(residual, valid, ys, condition.endswith('balanced'))
        tw = torch.tensor(weights, device='cuda', dtype=torch.float32)
        inputs = torch.zeros_like(tx) if body_only else tx
        torch.manual_seed(plan['seed'])
        model = RegisteredPreview().to('cuda')
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            digest.update(name.encode('utf8'))
            digest.update(parameter.detach().cpu().numpy().tobytes())
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                      weight_decay=plan['weight_decay'])
        losses = []
        for epoch in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(inputs, tb)-ty).square()*tw).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite residual loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite residual gradient')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if epoch % 25 == 0:
                write_json(run/'progress.json', dict(condition=condition, epoch=epoch+1,
                                                     loss=losses[-1]))
        model.eval()
        predictions = {}
        for e in episodes:
            base = ((e['x']['registered']-xm)/xs).clip(-5, 5).astype(np.float32)
            variants = {'intact': np.zeros_like(base) if body_only else base}
            if not body_only:
                changed = base.copy()
                changed[:, :896] = 0
                variants['neural_feature_mean'] = changed
                raw = e['x']['registered'].copy()
                raw[:, :896] = 0
                variants['retinal_neural_mean'] = ((raw-xm)/xs).clip(-5, 5).astype(np.float32)
                for view in ('front', 'left', 'right', 'rear'):
                    raw = e['x']['registered_'+view+'_mean']
                    variants[view+'_neural_mean'] = ((raw-xm)/xs).clip(-5, 5).astype(np.float32)
            variants['prior_only'] = None
            for intervention, values in variants.items():
                if values is None:
                    inferred = e['prior'].copy()
                else:
                    with torch.no_grad():
                        output = model(torch.tensor(values, device='cuda'),
                                       torch.tensor(e['b'], device='cuda')).cpu().numpy()
                    inferred = output*ys+rm+e['prior']
                predictions[e['id']+'__'+intervention] = inferred
                head_mse = [np.mean(((e['y'][e['valid'][:, h], h]-
                    inferred[e['valid'][:, h], h])/ys[h])**2) for h in range(6)]
                available = e['available'] & e['valid'][:, 5]
                groups = anticipation_groups(e['prior'][:, 5], e['y'][:, 5],
                                             available, e['valid'][:, 5])
                grouped = {}
                for name, keep in dict(full_valid=e['valid'][:, 5],
                                       fair_body_rows=available, **groups).items():
                    grouped[name] = dict(count=int(keep.sum()),
                        rmse=float(np.sqrt(np.mean((inferred[keep, 5]-e['y'][keep, 5])**2)))
                        if keep.any() and name != 'invalid_target' else None)
                reports.append(dict(condition=condition, episode=e['id'], role=e['role'],
                    intervention=intervention, standardized_mse=float(np.mean(head_mse)),
                    unavailable_prior_rows=int((~e['available']).sum()), groups20=grouped,
                    **metrics(e['y'], e['valid'], e['clock'], inferred, ym)))
        np.savez_compressed(run/(condition+'-predictions.npz'), **predictions)
        torch.save(dict(model=model.state_dict(), plan=plan, condition=condition,
            feature_mean=xm, feature_scale=xs, body_mean=bm, body_scale=bs,
            residual_mean=rm, target_scale=ys, actuator_policy=False), run/(condition+'.pt'))
        records.append(dict(condition=condition, epochs=200, losses=losses,
            initial_parameter_sha256=digest.hexdigest(), residual_class_counts=counts.tolist(),
            parameter_count=sum(p.numel() for p in model.parameters())))
    write_json(run/'result.json', dict(status='completed_registered_turn_residual',
        records=records, reports=reports, optimizer_updates=800, train_target_scale=ys.tolist(),
        train_target_mean=ym.tolist(), residual_mean=rm.tolist(),
        author_weights_changed=False, flight_motor_transfer=False, actor_deployed=False,
        goal_achieved=False, independent_final_validation=False))


if __name__ == '__main__':
    main()
