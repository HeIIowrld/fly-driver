"""Audit neutral temporal features on saved FlyVis edge responses; no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_temporal_motion_boundary import TemporalMotionBoundary
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'source', 'qualification', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    training, source, qualification = map(Path, (a.training, a.source, a.qualification))
    tr, saved, qr = (read(training/'result.json'), read(source/'result.json'),
                     read(qualification/'result.json'))
    qc = read(qualification/'manifest.json')['config']
    stimulus = read(source/'stimulus.json')
    boundary_source = 'src/flybrain_drive/flight_temporal_motion_boundary.py'
    if (tr['status'] != 'completed_fixed_joint_command_training'
            or sha256(tr['checkpoint']) != tr['checkpoint_sha256']
            or saved['status'] != 'completed'
            or sha256(source/'responses.npz') != saved['responses_sha256']
            or qr['status'] != 'qualified_neutral_temporal_motion_graph'
            or qc['training_checkpoint_sha256'] != tr['checkpoint_sha256']
            or qc['boundary_source_sha256'] != sha256(boundary_source)
            or stimulus['mode'] != 'edge' or stimulus['dt'] != .005):
        raise ValueError('Exact original model, saved edge data and qualification required')
    cp = torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model']
    mean, scale = cp['backbone.scene_command.mean'], cp['backbone.scene_command.scale']
    boundary = TemporalMotionBoundary(mean, scale, 42).eval()
    torch.set_num_threads(1)
    params = stimulus['parameters']
    pairs = [(contrast, angle) for contrast in (0, 1) for angle in (0, 90, 180, 270)]
    speeds = (4.8, 13., 19.)
    with load_array(source/'responses.npz', allow_pickle=False) as raw:
        names = raw['central_cell_types'].tolist()
        channels = [names.index(f'{kind}{direction}') for kind in ('T4', 'T5')
                    for direction in 'abcd']
        response, lengths = raw['activity'].copy(), raw['valid_frames'].copy()
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'checkpoint_sha256': tr['checkpoint_sha256'], 'source_responses_sha256':
            saved['responses_sha256'], 'stimulus_sha256': sha256(source/'stimulus.json'),
        'qualification_result_sha256': sha256(qualification/'result.json'),
        'boundary_source_sha256': sha256(boundary_source), 'speeds': list(speeds),
        'directions': [0, 90, 180, 270], 'contrasts': [0, 1], 'dt_seconds': .005,
        'spatial_assumption': 'central cell replicated into 16 bins in one view',
        'other_views': 'TRAIN mean, not simultaneous natural scenes',
        'actor_deployed': False, 'optimizer_updates': 0})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports, output = [], {}
    for speed in speeds:
        selected = []
        for contrast, angle in pairs:
            match = [i for i, row in enumerate(params)
                     if row['intensity'] == contrast and row['angle'] == angle
                     and row['speed'] == speed]
            if len(match) != 1:
                raise ValueError('One exact saved stimulus per condition required')
            selected.append(match[0])
        length = min(int(lengths[i]) for i in selected)
        activity = response[selected, :length, :][:, :, channels]
        if length < 2 or not np.isfinite(activity).all():
            raise ValueError('Finite saved actual response histories required')
        state, traces, levels, clips = None, [], [], []
        view_valid = torch.ones((32, 4), dtype=torch.bool)
        with torch.no_grad():
            for frame in range(length):
                scene = mean.expand(32, -1).clone().reshape(32, 4, 9, 16)
                for condition in range(8):
                    for view in range(4):
                        scene[condition*4+view, view, 1:, :] = torch.tensor(
                            np.tanh(activity[condition, frame]), dtype=mean.dtype)[:, None]
                scene = scene.flatten(1)
                standardized = ((scene - mean)/scale)[:, boundary.motion_mask]
                delta, state = boundary.step(scene, view_valid, state, None if frame == 0 else .005)
                if torch.count_nonzero(delta):
                    raise ValueError('Zero projection must remain neutral')
                trace = (state[0]-state[1]).reshape(8, 4, 4, 8, 16)
                level = standardized.clamp(-5, 5).reshape(8, 4, 4, 8, 16)
                clipping = (standardized.abs() > 5).reshape(8, 4, 4, 8, 16)
                idx = torch.arange(4)
                traces.append(trace[:, idx, idx].numpy())
                levels.append(level[:, idx, idx].numpy())
                clips.append(clipping[:, idx, idx].numpy())
        traces, levels, clips = np.stack(traces), np.stack(levels), np.stack(clips)
        output[f'speed_{speed:g}_traces'] = traces
        output[f'speed_{speed:g}_levels'] = levels
        summary = {}
        for intensity in (0, 1):
            for first, opposite in ((0, 2), (1, 3)):
                ia, ib = intensity*4+first, intensity*4+opposite
                diff_trace = traces[:, ia] - traces[:, ib]
                diff_level = levels[:, ia] - levels[:, ib]
                summary[f'{intensity}_{pairs[ia][1]}_{pairs[ib][1]}'] = {
                    'trace_rms_per_view': np.sqrt(np.mean(diff_trace**2, axis=(0, 2, 3))).tolist(),
                    'level_rms_per_view': np.sqrt(np.mean(diff_level**2, axis=(0, 2, 3))).tolist()}
        reports.append({'speed': speed, 'frames': length, 'pairs': summary,
                        'clipped_manipulated_motion_fraction': float(clips.mean())})
    np.savez_compressed(run/'temporal-features.npz', **output)
    result = {'status': 'completed_temporal_boundary_saved_edge_audit', 'reports': reports,
        'features_sha256': sha256(run/'temporal-features.npz'),
        'projection_is_zero_no_control_effect': True,
        'spatial_replication_is_synthetic': True, 'physiology_verified': False,
        'actor_deployed': False, 'optimizer_updates': 0, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
