"""Compare original image grids with body-registered neural fields on a virtual plane."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.body_retinal_plane import (
    fuse_scalar_views,
    plane_intersections,
    pool_on_plane,
)
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import activity, read
from scripts.train_visual_body_preview import direct_body_reference


def centres(nx, ny):
    x, y = np.meshgrid((np.arange(nx)+.5)*48/nx-24,
                       (np.arange(ny)+.5)*24/ny-12, indexing='ij')
    return np.column_stack([x.ravel(), y.ravel()])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan = read(a.plan)
    for f, digest in plan['source_sha256'].items():
        if sha256(f) != digest:
            raise ValueError('Pinned geometry preprocessing changed')
    reference, replay = Path(plan['reference_data']), Path(plan['sensor_replay'])
    source = read(reference/'dataset.json')
    replay_result = read(replay/'result.json')
    if replay_result['status'] != 'completed_spatial_flight_sensor_replay':
        raise ValueError('Every complete original neural history required')
    originals = {e['id']: e for e in replay_result['episodes']}
    if {e['id']: e['role'] for e in source['episodes']} != {
            k: e['role'] for k, e in originals.items()}:
        raise ValueError('Original whole roles required')
    cal = reference/'channel-calibration.npz'
    if sha256(cal) != source['channel_calibration_sha256']:
        raise ValueError('Unchanged TRAIN-only neural channel calibration required')
    with load_array(cal, allow_pickle=False) as arrays:
        mean, scale = arrays['processed_actual_mean'], arrays['processed_actual_scale']
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU neural-field pooling required')
    run = begin_run(a.output, dict(plan=plan, plan_sha256=sha256(a.plan),
        program_sha256=sha256(__file__), reference_dataset_sha256=sha256(reference/'dataset.json'),
        replay_result_sha256=sha256(replay/'result.json'), channel_calibration_sha256=sha256(cal),
        body_pose_is_allowed_actor_signal=True, measured_ground_or_depth=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    grid64, grid16 = (torch.tensor(centres(n, n), device='cuda', dtype=torch.float32)
                      for n in (8, 4))
    records, coverage = [], []
    for entry in source['episodes']:
        original = originals[entry['id']]
        data, clock = activity(original, read(replay/'manifest.json')['config'])
        body, body_valid, _ = direct_body_reference(entry, clock, sha256(plan['body_contract']))
        if sha256(entry['features_file']) != entry['features_sha256']:
            raise ValueError('Original image-grid reference changed')
        with load_array(entry['features_file'], allow_pickle=False) as old:
            np.testing.assert_array_equal(clock, old['observation_time'])
            baseline = old['processed_actual'].copy()
        registration = Path(original['output'])/'retina-registration.npz'
        with load_array(registration, allow_pickle=False) as reg:
            rays, origins = reg['body_directions'], reg['body_origins_m']
            if reg['view_order'].tolist() != ['front', 'left', 'right', 'rear']:
                raise ValueError('Original camera identities required')
        features = {'image_grid': np.concatenate([
            baseline, np.ones((len(clock), 128), np.float32)], 1)}
        for v, name in enumerate(('front', 'left', 'right', 'rear')):
            values = features['image_grid'].copy()
            values[:, v*224:(v+1)*224] = 0
            features['image_grid_'+name+'_mean'] = values
        for level in (False, True):
            key = 'registered_level' if level else 'registered'
            pitch, roll = (np.zeros(len(clock)), np.zeros(len(clock))) if level else (
                body[:, 0], body[:, 1])
            xy, supported = plane_intersections(rays, origins, pitch, roll,
                                                body_valid[:, 0] & body_valid[:, 1])
            outputs = {key: []}
            if not level:
                outputs.update({key+'_'+n+'_mean': [] for n in ('front', 'left', 'right', 'rear')})
            support64, support16 = [], []
            for start in range(0, len(clock), 16):
                end = min(len(clock), start+16)
                values = torch.tensor(data['processed_actual'][start:end], device='cuda')
                m = torch.tensor(mean, device='cuda', dtype=torch.float32)[None, None, :, None]
                s = torch.tensor(scale, device='cuda', dtype=torch.float32)[None, None, :, None]
                values = torch.tanh((values-m)/s)
                coords = torch.tensor(xy[start:end], device='cuda', dtype=torch.float32)
                mask = torch.tensor(supported[start:end], device='cuda')
                scalar, scalar_mask = pool_on_plane(values[:, :, :6], coords, mask, grid64,
                    torch.tensor([4., 2.], device='cuda'))
                motion, motion_mask = pool_on_plane(values[:, :, 6:], coords, mask, grid16,
                    torch.tensor([8., 4.], device='cuda'))
                fused, fused_mask = fuse_scalar_views(scalar, scalar_mask)
                flags = torch.cat([fused_mask, motion_mask.flatten(1)], 1).float()

                def packed(f, flow, flags=flags):
                    return torch.cat([f.flatten(1), flow.flatten(1), flags], 1).cpu().numpy()

                outputs[key].append(packed(fused, motion))
                if not level:
                    count = scalar_mask.sum(1).clamp_min(1)[:, None]
                    for v, name in enumerate(('front', 'left', 'right', 'rear')):
                        changed = fused-scalar[:, v]*scalar_mask[:, v, None]/count
                        flow = motion.clone()
                        flow[:, v] = 0
                        outputs[key+'_'+name+'_mean'].append(packed(changed, flow))
                support64.append(fused_mask.cpu().numpy())
                support16.append(motion_mask.cpu().numpy())
            features.update({k: np.concatenate(v) for k, v in outputs.items()})
            coverage.append(dict(id=entry['id'], role=entry['role'], condition=key,
                ray_support_fraction_by_view=supported.mean(axis=(0, 2)).tolist(),
                scalar_bin_fraction=np.concatenate(support64).mean(0).tolist(),
                motion_bin_fraction_by_view=np.concatenate(support16).mean((0, 2)).tolist()))
        path = run/(entry['id']+'.npz')
        np.savez_compressed(path, **features, observation_time=clock)
        records.append(dict(entry, features_file=str(path), features_sha256=sha256(path)))
        write_json(run/'progress.json', dict(completed=len(records), total=4))
    write_json(run/'dataset.json', dict(
        schema='flybrain.body_registered_preview.v1', episodes=records,
        conditions=['image_grid', 'registered', 'registered_level'], feature_count=1024,
        neural_feature_count=896, support_feature_count=128, measured_road_visibility=False,
        scalar_layout='six neuron types fused equally over supported cameras,8x8 common body grid',
        motion_layout=('eight original direction types per original camera,4x4 body grid; '
                       'no directional-type rotation or fusion'),
        support_is_pose_dependent=True, original_targets_clocks_roles_unchanged=True))
    write_json(run/'result.json', dict(status='completed_body_registered_preview_data',
        coverage=coverage, optimizer_updates=0, author_weights_changed=False,
        actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
