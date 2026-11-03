"""New-mount-only visual preview dataset; TRAIN calibration and full fixed-role histories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_camera_mount import validate_frame_mount
from flybrain_drive.ac_csp_geometry import matrix
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_feature_scaling import ChannelMoments
from flybrain_drive.teacher_path_preview import future_path_preview
from scripts.build_retinotopic_preview_dataset import activity, read
from scripts.replay_spatial_flight_sensors import verify_roles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan = read(a.plan)
    for source, digest in plan['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Pinned dataset source changed')
    replay = Path(plan['sensor_replay'])
    result, provenance = read(replay/'result.json'), read(replay/'manifest.json')['config']
    collection, audit = Path(plan['collection']), Path(plan['audits'])
    terminal, audited = read(collection/'result.json'), read(audit/'result.json')
    verify_roles(terminal, audited)
    if (result['status'] != 'completed_spatial_flight_sensor_replay'
            or provenance['audit_result_sha256'] != sha256(audit/'result.json')
            or {e['recording']: e['role'] for e in result['episodes']} != {
                e['path']: e['role'] for e in audited['episodes']}):
        raise ValueError('Every audited original sensor replay and role required')
    camera_config = read(plan['camera_config'])
    calibration = {k: ChannelMoments() for k in ('raw_control', 'processed_actual')}
    for entry in result['episodes']:
        data, _ = activity(entry, dict(plan=plan))
        if entry['role'] == 'train':
            for key, values in data.items():
                calibration[key].add(values, role='train')
    moments = {k: v.finish(plan['channel_scale_floor']) for k, v in calibration.items()}
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU pooling required')
    first = Path(result['episodes'][0]['output'])
    with load_array(first/'retina-registration.npz', allow_pickle=False) as reg:
        registered = {k: reg[k].copy() for k in ('image_uv', 'body_directions', 'view_order',
                                                'motion_types', 'body_origins_m')}
    pool = FlightScenePool(registered['image_uv'], read(plan['scene_spec'])['scene_basis']).to('cuda')
    run = begin_run(a.output, dict(plan=plan, plan_sha256=sha256(a.plan),
        program_sha256=sha256(__file__), replay_result_sha256=sha256(replay/'result.json'),
        actor_oracle_inputs=False, fitting_ids=[e['id'] for e in result['episodes']
                                               if e['role'] == 'train']))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'channel-calibration.npz', **{k+'_'+label: value
        for k, pair in moments.items() for label, value in zip(('mean', 'scale'), pair, strict=True)})
    records = []
    for entry in result['episodes']:
        root, vision = Path(entry['recording']), Path(entry['output'])
        if (read(vision/'manifest.json')['config']['events_sha256'] != sha256(root/'events.jsonl')
                or sha256(vision/'result.json') != entry['result_sha256']
                or read(root/'manifest.json')['config']['sensory_configuration'] != camera_config):
            raise ValueError('Original camera and event/replay identity required')
        with load_array(vision/'retina-registration.npz', allow_pickle=False) as reg:
            for key, value in registered.items():
                np.testing.assert_array_equal(reg[key], value)
        data, clocks = activity(entry, dict(plan=plan))
        features = {}
        for key, values in data.items():
            mean, scale = moments[key]
            with torch.no_grad():
                x = torch.tensor(values, device='cuda', dtype=torch.float32)
                m = torch.tensor(mean, device='cuda', dtype=torch.float32)[None, None, :, None]
                s = torch.tensor(scale, device='cuda', dtype=torch.float32)[None, None, :, None]
                features[key] = (torch.tanh((x-m)/s) @ pool.weight.T).flatten(1).cpu().numpy()
        events = [json.loads(line) for line in (root/'events.jsonl').read_text('utf8').splitlines()]
        np.testing.assert_array_equal(clocks, [e['decision_monotonic_seconds'] for e in events])
        telemetry = read(root/'control/api-receipt.json')['telemetry']
        targets, masks = [], []
        with CameraArchive(root/'vision') as archive:
            for event in events:
                raw = archive.read(Path(event['camera']['file']).name)
                if hashlib.sha256(raw).hexdigest() != event['camera']['file_sha256']:
                    raise ValueError('Original camera metadata differs')
                frame = json.loads(raw)
                validate_frame_mount(frame, camera_config)
                front = next(v for v in frame['views'] if v['view_id'] == 'front')
                target, mask = future_path_preview(telemetry, front['render_started_clock'],
                    front['renderer_only_world_look'], matrix(front['view_matrix_diagnostic'])[:3, 0])
                targets.append(target)
                masks.append(mask)
        path = run/(entry['id']+'.npz')
        np.savez_compressed(path, **features, preview=np.array(targets),
                            preview_valid=np.array(masks), observation_time=clocks)
        records.append(dict(id=entry['id'], role=entry['role'], group='new_exterior',
            features_file=str(path), targets_file=str(path), features_sha256=sha256(path),
            targets_sha256=sha256(path), body_replay=entry['body_replay'],
            body_result_sha256=entry['body_result_sha256'], observations=len(clocks),
            valid_per_distance=np.array(masks).sum(0).tolist()))
        write_json(run/'progress.json', dict(completed=len(records), total=4))
    write_json(run/'dataset.json', dict(schema='flybrain.retinotopic_preview.episodes.v1',
        episodes=records, conditions=['raw_control', 'processed_actual'],
        channel_calibration_sha256=sha256(run/'channel-calibration.npz'),
        channel_calibration_file=str(run/'channel-calibration.npz'),
        processed_channels=plan['types']+['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'],
        raw_control_channels=['retina_intensity']+['constant_zero']*5
            +['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'],
        original_targets_roles_clocks_unchanged=True, body_or_oracle_inputs_added=False,
        camera_mount_contract=camera_config['camera_mount_contract'],
        target_reference='vehicle-origin future travelled path in horizontal front-camera basis',
        targets_are_optimal_racing_line=False, calibrate_only_train=True))
    write_json(run/'result.json', dict(status='completed_exterior_flight_preview_dataset',
        episodes=len(records), optimizer_updates=0, author_weights_changed=False,
        target_clipping=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
