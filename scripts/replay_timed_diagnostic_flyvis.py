"""Replay diagnostic retinal groups through frozen FlyVis; no actor or optimizer."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER, integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projection', required=True)
    p.add_argument('--recording', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--online-shadow', action='store_true',
                   help='Replay at recorded online body selection host clocks')
    p.add_argument('--visual-snapshot', action='store_true',
                   help='Replay at original visual snapshot rather than body selection')
    args = p.parse_args()
    if args.visual_snapshot and not args.online_shadow:
        raise ValueError('Visual snapshot requires the original online shadow')
    root, projection = Path(args.recording), Path(args.projection)
    source = json.loads((root/'result.json').read_text(encoding='utf8'))
    projected = json.loads((projection/'result.json').read_text(encoding='utf8'))
    projected_config = json.loads((projection/'manifest.json').read_text(encoding='utf8'))['config']
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    if (source.get('status') != 'completed' or source.get('training_admissible') is not False
            or projected.get('status') != 'completed_retina_body_diagnostic_projection'
            or projected.get('training_admissible') is not False
            or projected_config['recording_result_sha256'] != sha256(root/'result.json')
            or sha256(projection/'retina-body-diagnostic.npz') != projected['arrays_sha256']
            or sha256(config['author_model_dir']+'/best_chkpt') != config['author_weights_sha256']
            or sha256(config['retina_registration']) != config['retina_registration_sha256']):
        raise ValueError('Exact diagnostic projection and pinned author model required')
    rows = json.loads((root/'diagnostic-frames.json').read_text(encoding='utf8'))
    if sha256(root/'diagnostic-frames.json') != source['metadata_sha256']:
        raise ValueError('Original camera metadata differs')
    availability = []
    for index, row in enumerate(rows):
        file = Path(row['source_file'])
        if row['index'] != index or sha256(file) != row['sha256']:
            raise ValueError('Original indexed camera frame differs')
        frame = json.loads(file.read_text(encoding='utf8'))
        views = {v['view_id']: v for v in frame['views']}
        if set(views) != set(VIEW_ORDER):
            raise ValueError('All four original views required')
        # Producer save completion is a conservative group availability time, not exposure.
        availability.append(max(views[name]['save_finished_clock'] for name in VIEW_ORDER))
    availability = np.asarray(availability)
    clock_source = 'group latest original source save clock'
    if args.online_shadow:
        if (source.get('shadow_error') is not None
                or source.get('shadow_sha256') != sha256(root/'online-body-shadow.json')):
            raise ValueError('Exact completed original online shadow required')
        events = json.loads((root/'online-body-shadow.json').read_text(encoding='utf8'))
        if (len(events) != len(rows) or any(event['index'] != i
                or event['metadata_sha256'] != rows[i]['sha256']
                or event['decision'] < event['camera_receipt'] for i, event in enumerate(events))):
            raise ValueError('Original online decision and frame lineage required')
        if args.visual_snapshot:
            if any('visual_snapshot' not in event or 'vision_finished' not in event['neural']
                    or not event['camera_receipt'] <= event['visual_snapshot']
                    <= event['neural']['vision_finished'] <= event['decision'] for event in events):
                raise ValueError('Exact original visual and later decision clock ordering required')
            availability = np.asarray([event['visual_snapshot'] for event in events], dtype=np.float64)
            clock_source = 'original host visual snapshot preceding actual visual completion'
        else:
            availability = np.asarray([event['decision'] for event in events], dtype=np.float64)
            clock_source = 'original host body-selection snapshot, not actual neural inference'
    with np.load(projection/'retina-body-diagnostic.npz', allow_pickle=False) as data:
        retina, body_valid = data['retinal'].copy(), data['body_valid'].copy()
    if (retina.shape != (len(rows), 4, 721) or not np.isfinite(retina).all()
            or body_valid.shape != (len(rows), 4, 21) or np.any(np.diff(availability) <= 0)
            or np.any(np.diff(availability) > .15)):
        raise ValueError('Complete finite original groups with bounded ordered clocks required')
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no fallback')
    os.environ['FLYVIS_ROOT_DIR'] = str(Path(config['cache_root']).resolve())
    os.environ['MPLBACKEND'] = 'Agg'
    model = Path(config['author_model_dir'])
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'projection_result_sha256': sha256(projection/'result.json'),
        'retina_sha256': projected['arrays_sha256'],
        'recording_result_sha256': sha256(root/'result.json'),
        'weights_sha256': config['author_weights_sha256'], 'seed': 863009,
        'timing': 'hold prior group until declared group availability; not same-time exposure',
        'clock_source': clock_source,
        'online_shadow_sha256': source.get('shadow_sha256') if args.online_shadow else None,
        'actor_input': False, 'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'actor_deployed': False, 'optimizer_updates': 0,
              'training_admissible': False, 'physical_sample_or_exposure_time_verified': False}
    try:
        install_datamate_cache_compat(config['cache_root'])
        import flyvis

        from flybrain_drive.flyvis_fractional_tail import FractionalVisualTail

        torch.set_default_device('cuda')
        author = yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))
        network = flyvis.Network(**author['config']['network']).to('cuda')
        weights = torch.load(model/'best_chkpt', map_location='cpu', weights_only=True)['network']
        network.load_state_dict(weights, strict=True)
        network.eval().requires_grad_(False)
        nodes = network.connectome.nodes
        receptors = network.stimulus.input_index[0]
        u, v = nodes.u[:][receptors], nodes.v[:][receptors]
        with load_array(config['retina_registration'], allow_pickle=False) as reg:
            np.testing.assert_array_equal(u, reg['u'])
            np.testing.assert_array_equal(v, reg['v'])
            saved_indices = reg['motion_indices'].copy()
            saved_types = reg['motion_types'].astype(str).copy()
        node_types = nodes.type[:].astype(str)
        all_u, all_v = nodes.u[:], nodes.v[:]
        indices = []
        for name in saved_types:
            mapping = {(int(all_u[i]), int(all_v[i])): i
                       for i in np.flatnonzero(node_types == name)}
            indices.append([mapping[(int(q), int(r))] for q, r in zip(u, v, strict=True)])
        np.testing.assert_array_equal(indices, saved_indices)
        central = network.connectome.central_cells_index[:]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        motion, central_activity = [], []
        ticks = 0
        with torch.no_grad():
            state = network.steady_state(1., .005, batch_size=4, value=.5)
            network.simulate = FractionalVisualTail(network, state).simulate
            held = None
            for i, timestamp in enumerate(availability):
                if i:
                    for dt in integration_intervals(float(timestamp-availability[i-1])):
                        state = network.simulate(held[:, None, None, :], dt=dt,
                                                 initial_state=state, as_states=True)[-1]
                        ticks += 1
                if not torch.isfinite(state.nodes.activity).all():
                    raise ValueError('Finite uninterrupted neural state required')
                motion.append(state.nodes.activity[:, saved_indices].cpu().numpy())
                central_activity.append(state.nodes.activity[:, central].cpu().numpy())
                held = torch.as_tensor(retina[i], dtype=torch.float32, device='cuda')
                if i % 25 == 0:
                    write_json(run/'progress.json', {'completed_groups': i+1,
                        'groups': len(rows), 'ticks': ticks})
        torch.cuda.synchronize()
        if any(not torch.equal(value.cpu(), weights[key].cpu())
               for key, value in network.state_dict().items() if key in weights):
            raise ValueError('Frozen author network weights changed')
        np.savez_compressed(run/'responses.npz', motion_activity=np.stack(motion),
            central_activity=np.stack(central_activity), retinal_intensity=retina,
            group_source_available_clock=availability, body_valid=body_valid,
            motion_types=saved_types)
        result.update(status='completed_diagnostic_flyvis_replay', groups=len(rows),
            clock_source=clock_source,
            motion_shape=list(np.stack(motion).shape), internal_ticks=ticks,
            independent_view_states=4, original_group_resets=0,
            weights_changed=False, future_group_used_for_previous_interval=False,
            elapsed_seconds=time.perf_counter()-started,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            responses_sha256=sha256(run/'responses.npz'),
            actual_fly_physiology_or_optics_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
