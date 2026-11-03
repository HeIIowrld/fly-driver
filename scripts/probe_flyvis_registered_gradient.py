"""Check a real author-network gradient through registered vision; zero optimizer updates."""

import argparse
import inspect
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.body_retinal_plane import (
    fuse_scalar_views,
    plane_intersections,
    pool_on_plane,
)
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from scripts.build_body_registered_preview import centres
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_visual_body_preview import body_features, direct_body_reference


def state_tensors(value, prefix='state'):
    """Inspect stored state, including RefTensor indices, without populating caches."""
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        leaves = {}
        for key in value:
            leaves.update(state_tensors(dict.__getitem__(value, key), prefix+'.'+str(key)))
        return leaves
    if hasattr(value, 'values') and hasattr(value, 'indices'):
        return {prefix+'.values': value.values, prefix+'.indices': value.indices}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {}
    raise TypeError('Uninspected recurrent state type: '+str(type(value)))


def detached_state_exact(state):
    before = {key: value.detach().clone() for key, value in state_tensors(state).items()}
    detached = state.detach()
    after = state_tensors(detached)
    if before.keys() != after.keys() or not before:
        raise ValueError('Recurrent state structure changed during detach')
    for key, value in after.items():
        if value.requires_grad or not torch.equal(before[key], value):
            raise ValueError('Detach changed a state value or retained its gradient history')
    return detached, len(before)


def registered_features(selected, rays, origins, body, body_valid, mean, scale):
    m = torch.tensor(mean, dtype=torch.float32)[None, :, None]
    s = torch.tensor(scale, dtype=torch.float32)[None, :, None]
    normalized = torch.tanh((selected-m)/s)[None]
    xy, supported = plane_intersections(rays, origins, body[:, 0], body[:, 1],
                                        body_valid[:, 0] & body_valid[:, 1])
    xy = torch.tensor(xy, dtype=torch.float32)
    supported = torch.tensor(supported)
    scalar, scalar_mask = pool_on_plane(normalized[:, :, :6], xy, supported,
        torch.tensor(centres(8, 8), dtype=torch.float32), torch.tensor([4., 2.]))
    flow, flow_mask = pool_on_plane(normalized[:, :, 6:], xy, supported,
        torch.tensor(centres(4, 4), dtype=torch.float32), torch.tensor([8., 4.]))
    fused, fused_mask = fuse_scalar_views(scalar, scalar_mask)
    return torch.cat([fused.flatten(1), flow.flatten(1), fused_mask.float(),
                      flow_mask.flatten(1).float()], dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered-data', required=True)
    p.add_argument('--training', required=True)
    p.add_argument('--sensor-replay', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--intervals', type=int, default=2)
    p.add_argument('--chunk-intervals', type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.chunk_intervals <= a.intervals <= 128:
        raise ValueError('Explicit bounded original TRAIN prefix and chunk length required')
    data, training, replay = (Path(x) for x in (a.registered_data, a.training, a.sensor_replay))
    dataset = read(data/'dataset.json')
    entry = next(e for e in dataset['episodes'] if e['role'] == 'train')
    original = next(e for e in read(replay/'result.json')['episodes'] if e['id'] == entry['id'])
    vision, model_dir = Path(original['output']), Path(a.model_dir)
    if (read(training/'manifest.json')['config']['dataset_sha256'] != sha256(data/'dataset.json')
            or sha256(vision/'responses.npz') != original['responses_sha256']):
        raise ValueError('Exact TRAIN history and registered readout required')
    checkpoint = training/'registered.pt'
    run = begin_run(a.output, dict(arguments=vars(a), program_sha256=sha256(__file__),
        dataset_sha256=sha256(data/'dataset.json'), episode=entry['id'], role='train',
        author_checkpoint_sha256=sha256(model_dir/'best_chkpt'),
        readout_checkpoint_sha256=sha256(checkpoint), decision_index=a.intervals,
        chunk_intervals=a.chunk_intervals,
        optimizer_updates=0, original_trainable_flags_preserved=True,
        implementation='Author _param_api/_next_state; clamp before trace, no frozen CUDA graph'))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False,
                  goal_achieved=False, physiological_function_verified=False)
    try:
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required')
        os.environ['FLYVIS_ROOT_DIR'] = str(Path(a.cache_root).resolve())
        os.environ['MPLBACKEND'] = 'Agg'
        install_datamate_cache_compat(a.cache_root)
        import flyvis

        torch.set_default_device('cuda')
        config = yaml.safe_load((model_dir/'_meta.yaml').read_text(encoding='utf8'))
        network = flyvis.Network(**config['config']['network']).to('cuda')
        original_flags = {name: p.requires_grad for name, p in network.named_parameters()}
        implementation = Path(inspect.getsourcefile(type(network)))
        write_json(run/'author-implementation.json', dict(source=str(implementation),
            source_sha256=sha256(implementation), model_meta_sha256=sha256(model_dir/'_meta.yaml'),
            original_trainable_flags=original_flags))
        weights = torch.load(model_dir/'best_chkpt', map_location='cpu',
                             weights_only=True)['network']
        network.load_state_dict(weights, strict=True)
        network.eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            state = network.steady_state(1., .005, batch_size=4, value=.5)
        for name, parameter in network.named_parameters():
            parameter.requires_grad_(original_flags[name])
        with load_array(vision/'responses.npz', allow_pickle=False) as arrays:
            times = arrays['decision_monotonic_seconds'][:a.intervals+1].copy()
            retinas = arrays['retina_intensity'][:a.intervals].copy()
            expected = np.concatenate([arrays['additional_activity'][:a.intervals+1],
                                        arrays['motion_activity'][:a.intervals+1]], axis=2)
        with load_array(vision/'retina-registration.npz', allow_pickle=False) as reg:
            indices = np.concatenate([reg['additional_indices'], reg['motion_indices']], axis=0)
            rays, origins = reg['body_directions'].copy(), reg['body_origins_m'].copy()
        # The author's public forward clamps once per sequence. Reuse its exact
        # parameter API for variable original dt; do not mutate parameter versions
        # between differentiable steps or reuse mutable stimulus-buffer views.
        network.clamp()
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        readout = RegisteredPreview().to('cuda').eval().requires_grad_(False)
        readout.load_state_dict(saved['model'])
        with load_array(entry['targets_file'], allow_pickle=False) as arrays:
            clock = arrays['observation_time']
            targets = arrays['preview'][:a.intervals+1].reshape(-1, 6).copy()
            validity = np.repeat(arrays['preview_valid'][:a.intervals+1], 2, axis=1)
        body, body_valid, _ = direct_body_reference(entry, clock,
            sha256('configs/ac-virtual-navigation-body-v2.json'))
        cal_path = Path(read(data/'manifest.json')['config']['plan']['reference_data'])
        with load_array(cal_path/'channel-calibration.npz', allow_pickle=False) as arrays:
            mean, scale = arrays['processed_actual_mean'], arrays['processed_actual_scale']
        with load_array(entry['features_file'], allow_pickle=False) as arrays:
            expected_features = arrays['registered'][:a.intervals+1].copy()
        ticks, neural_error, feature_error, chunks = 0, 0., 0., []
        for start in range(0, a.intervals, a.chunk_intervals):
            wall = time.perf_counter()
            network.zero_grad(set_to_none=True)
            # Derived parameter tensors belong to this chunk's autograd graph.
            params = network._param_api()
            end = min(start+a.chunk_intervals, a.intervals)
            for i in range(start, end):
                network.stimulus.zero(4, 1)
                network.stimulus.add_input(torch.tensor(retinas[i])[:, None, None, :])
                stimulus = network.stimulus()[:, 0].clone()
                for dt in integration_intervals(float(times[i+1]-times[i])):
                    state = network._next_state(params, state, stimulus, dt)
                    ticks += 1
                selected = state.nodes.activity[:, torch.tensor(indices, dtype=torch.long)]
                error = float(np.abs(selected.detach().cpu().numpy()-expected[i+1]).max())
                neural_error = max(neural_error, error)
                if error > 2e-6:
                    raise ValueError('Differentiable trace differs from frozen original history')
            features = registered_features(selected, rays, origins, body[end:end+1],
                body_valid[end:end+1], mean, scale)
            error = float(np.abs(features.detach().cpu().numpy()[0]-expected_features[end]).max())
            feature_error = max(feature_error, error)
            if error > 2e-6:
                raise ValueError('Differentiable features differ from original dataset')
            x = ((features-torch.tensor(saved['feature_mean'], dtype=torch.float32))/torch.tensor(
                saved['feature_scale'], dtype=torch.float32)).clamp(-5, 5)
            b = body_features(body[end:end+1], body_valid[end:end+1], saved['body_mean'],
                saved['body_scale'], saved['plan']['body_route_indices'])
            output = readout(x, torch.tensor(b))
            expected_target = torch.tensor(
                (targets[end:end+1]-saved['target_mean'])/saved['target_scale'],
                dtype=torch.float32)
            valid = validity[end]
            if not valid.any():
                raise ValueError('Declared chunk endpoint has no valid original target')
            loss = (output[:, valid]-expected_target[:, valid]).square().mean()
            if not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError('No finite gradient path into author network')
            loss.backward()
            chunk_gradients = []
            for name, parameter in network.named_parameters():
                if not original_flags[name] and parameter.grad is not None:
                    raise ValueError('Fixed author parameter received a chunk gradient')
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError('Nonfinite chunk gradient')
                    chunk_gradients.append(int(torch.count_nonzero(parameter.grad)))
            if not sum(chunk_gradients):
                raise ValueError('No nonzero gradient at a declared chunk endpoint')
            state, leaves = detached_state_exact(state)
            chunks.append(dict(start_decision=start, end_decision=end,
                loss=float(loss.detach()), nonzero_gradients=sum(chunk_gradients),
                detached_state_tensors=leaves, exact_state_detach=True,
                wall_seconds=time.perf_counter()-wall,
                allocated_gpu_bytes=torch.cuda.memory_allocated()))
            write_json(run/'progress.json', dict(completed_chunks=len(chunks),
                covered_intervals=end, internal_ticks=ticks, last_chunk=chunks[-1]))
        gradients = []
        for name, parameter in network.named_parameters():
            if not original_flags[name] and parameter.grad is not None:
                raise ValueError('Originally fixed author parameter received a gradient')
            if original_flags[name]:
                grad = parameter.grad
                if grad is not None and not torch.isfinite(grad).all():
                    raise ValueError('Nonfinite author gradient')
                gradients.append(dict(name=name, elements=parameter.numel(),
                    gradient_present=grad is not None,
                    nonzero=int(torch.count_nonzero(grad)) if grad is not None else 0,
                    norm=float(grad.norm()) if grad is not None else 0.))
        changed = [k for k, value in network.state_dict().items()
                   if k in weights and not torch.equal(value.detach().cpu(), weights[k].cpu())]
        if changed or not any(g['nonzero'] for g in gradients):
            raise ValueError('Author weights changed or no nonzero trainable-author gradient')
        result.update(status='completed_registered_author_gradient_probe',
            loss=float(loss.detach()), neural_error=neural_error, feature_error=feature_error,
            gradients=gradients, author_weights_changed=changed, internal_ticks=ticks,
            original_elapsed_seconds=float(times[-1]-times[0]), independent_camera_states=4,
            chunks=chunks, recurrent_numeric_resets=0, original_interval_count=a.intervals,
            frozen_readout_gradients=any(p.grad is not None for p in readout.parameters()),
            peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated(),
            scope='TRAIN prefix gradient preflight; no learning or motor qualification')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
