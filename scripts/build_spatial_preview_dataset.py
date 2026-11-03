"""Retain old roles and add all complete new TRAIN histories to a visual-only auxiliary task."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_geometry import camera_geometry, matrix
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from flybrain_drive.teacher_path_preview import future_path_preview


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def pool_replay(root, pool):
    report = read(root/'result.json')
    if (report['status'] != 'completed'
            or sha256(root/'responses.npz') != report['responses_sha256']):
        raise ValueError('Complete unchanged author replay required')
    with load_array(root/'responses.npz', allow_pickle=False) as arrays:
        scene = np.concatenate([arrays['retina_intensity'][:, :, None],
                                arrays['motion_activity']], axis=2)
        clocks = arrays['decision_monotonic_seconds'].copy()
    with torch.no_grad():
        features = pool(torch.tensor(scene, dtype=torch.float32, device='cuda')).cpu().numpy()
    return features, clocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    base, preview, audited, visual = (Path(plan[k]) for k in (
        'base_data', 'base_preview', 'spatial_audit', 'visual_replay'))
    original = read(base/'dataset.json')
    labels = {e['file']: e for e in read(preview/'result.json')['episodes']}
    new = read(audited/'result.json')
    if (new['status'] != 'completed_partial_spatial_history_raw_audit'
            or len(new['episodes']) != 2 or any(e['role'] != 'train' for e in new['episodes'])
            or any(e['role'] not in ('train', 'development') for e in original['episodes'])):
        raise ValueError('Original roles and both preassigned new TRAIN histories required')
    spec = read(plan['scene_spec'])
    if sha256(spec['scene_registration_file']) != spec['scene_registration_sha256']:
        raise ValueError('Canonical optical registration changed')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Same GPU pooling path required')
    with load_array(spec['scene_registration_file'], allow_pickle=False) as registration:
        registered = {k: registration[k].copy() for k in
                      ('image_uv', 'body_directions', 'view_order', 'motion_types')}
    pool = FlightScenePool(registered['image_uv'], spec['scene_basis']).to('cuda')
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan,
        program_sha256=sha256(__file__), plan_sha256=sha256(args.plan),
        base_dataset_sha256=sha256(base/'dataset.json'),
        spatial_audit_result_sha256=sha256(audited/'result.json'),
        label_module_sha256=sha256('src/flybrain_drive/teacher_path_preview.py'),
        scene_spec_sha256=sha256(plan['scene_spec']), device=torch.cuda.get_device_name(0)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    records = []
    for entry in original['episodes']:
        feature_file, target_file = base/entry['file'], preview/entry['file']
        if (sha256(feature_file) != entry['sha256']
                or sha256(target_file) != labels[entry['file']]['auxiliary_sha256']):
            raise ValueError('Original feature/auxiliary arrays changed')
        with load_array(feature_file, allow_pickle=False) as f, load_array(
                target_file, allow_pickle=False) as t:
            np.testing.assert_array_equal(f['observation_time'], t['observation_time'])
        records.append(dict(id='original-'+entry['file'].removesuffix('.npz'),
            role=entry['role'], group='original', features_file=str(feature_file),
            targets_file=str(target_file), features_sha256=sha256(feature_file),
            targets_sha256=sha256(target_file), observations=entry['observations']))
    # Reproduce one whole original TRAIN using the same GPU pool, before adding new inputs.
    reference = next(e for e in original['episodes'] if e['role'] == 'train')
    reproduced, clocks = pool_replay(Path(reference['original']['scene_replay']), pool)
    with load_array(base/reference['file'], allow_pickle=False) as expected:
        np.testing.assert_array_equal(clocks, expected['observation_time'])
        reproduction_error = float(np.abs(reproduced-expected['scene_features']).max())
    if reproduction_error > plan['pool_reproduction_tolerance']:
        raise ValueError('Original GPU scene pooling does not reproduce')
    for entry in new['episodes']:
        root = Path(entry['path'])
        replay = visual/root.name
        proof = read(Path(entry['raw_audit'])/'manifest.json')['config']
        if (sha256(Path(entry['raw_audit'])/'result.json') != entry['raw_audit_result_sha256']
                or proof['recording_manifest_sha256'] != sha256(root/'manifest.json')
                or proof['events_sha256'] != sha256(root/'events.jsonl')
                or read(replay/'manifest.json')['config']['events_sha256'] != (
                    proof['events_sha256'])):
            raise ValueError('Original raw audit and visual replay identity differ')
        with load_array(replay/'retina-registration.npz', allow_pickle=False) as reg:
            for key, value in registered.items():
                np.testing.assert_array_equal(reg[key], value)
        features, clocks = pool_replay(replay, pool)
        events = [json.loads(line) for line in (root/'events.jsonl').read_text('utf8').splitlines()]
        if len(events) != len(clocks):
            raise ValueError('Every original observation required')
        telemetry = read(root/'control/api-receipt.json')['telemetry']
        targets, masks = [], []
        with CameraArchive(root/'vision') as camera:
            for index, event in enumerate(events):
                if clocks[index] != event['decision_monotonic_seconds']:
                    raise ValueError('Original clock changed')
                raw = camera.read(Path(event['camera']['file']).name)
                if hashlib.sha256(raw).hexdigest() != event['camera']['file_sha256']:
                    raise ValueError('Original rendered metadata differs')
                frame = json.loads(raw)
                camera_geometry(frame['views'])
                front = next(v for v in frame['views'] if v['view_id'] == 'front')
                target, mask = future_path_preview(telemetry, front['render_started_clock'],
                    front['renderer_only_world_look'],
                    matrix(front['view_matrix_diagnostic'])[:3, 0])
                targets.append(target)
                masks.append(mask)
        path = run/(root.name+'.npz')
        np.savez_compressed(path, scene_features=features, preview=np.array(targets),
                            preview_valid=np.array(masks), observation_time=clocks)
        records.append(dict(id='spatial-'+root.name, role='train', group='new_spatial',
            features_file=str(path), targets_file=str(path), features_sha256=sha256(path),
            targets_sha256=sha256(path), observations=len(clocks),
            source_events_sha256=proof['events_sha256'], visual_responses_sha256=sha256(
                replay/'responses.npz'), valid_per_distance=np.array(masks).sum(0).tolist()))
        write_json(run/'progress.json', dict(new_histories=len(records)-len(original['episodes'])))
    write_json(run/'dataset.json', dict(schema='flybrain.visual_preview.episodes.v1',
        episodes=records,
        actor_inputs='scene576 only', targets='privileged future-path bearing and chord-curvature',
        distances_m=[5, 10, 20], source_collection_task_passed=False,
        old_roles_unchanged=True, failed_history_used=False,
        legacy_actor_goals_reinterpreted=False))
    write_json(run/'result.json', dict(status='completed_spatial_preview_dataset',
        episodes=len(records), new_train_histories=2, new_observations=1479,
        original_pool_reproduction_max_error=reproduction_error,
        original_payloads_referenced_not_replicated=True, target_clipping=False,
        optimizer_updates=0, student_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
