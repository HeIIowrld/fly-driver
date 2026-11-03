"""Offline signed port intervention on a contiguous observed-body diagnostic segment."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.graph import sha256


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('projection', 'replay', 'probe', 'training', 'opponent', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--double', action='store_true', help='Replay fixed float32 inputs in float64')
    parser.add_argument('--availability-body', help='Audited group availability body diagnostic')
    args = parser.parse_args()
    projection, replay, probe, training, opponent = map(
        Path, (args.projection, args.replay, args.probe, args.training, args.opponent))
    pr, rr, br, tr, op = (read_json(path/'result.json') for path in
                          (projection, replay, probe, training, opponent))
    manifest = read_json(probe/'manifest.json')['config']
    files = ((projection/'retina-body-diagnostic.npz', pr['arrays_sha256']),
             (replay/'responses.npz', rr['responses_sha256']),
             (probe/'port-probe.npz', br['arrays_sha256']),
             (opponent/'temporal-boundary.pt', op['checkpoint_sha256']))
    if (any(sha256(path) != expected for path, expected in files)
            or manifest['projection_sha256'] != pr['arrays_sha256']
            or manifest['flyvis_sha256'] != rr['responses_sha256']
            or manifest['temporal_checkpoint_sha256'] != op['checkpoint_sha256']
            or br['status'] != 'completed_offline_visual_body_port_probe'
            or br['training_admissible'] is not False):
        raise ValueError('Exact completed diagnostic lineage required')
    plan = read_json(training/'manifest.json')['config']['plan']
    source = read_json(Path(plan['source'])/'result.json')
    checkpoint = Path(tr['checkpoint'])
    if sha256(checkpoint) != op['source_checkpoint_sha256']:
        raise ValueError('Original temporal checkpoint parent required')
    availability_root = Path(args.availability_body) if args.availability_body else None
    availability_result = None
    if availability_root:
        availability_result = read_json(availability_root/'result.json')
        availability_manifest = read_json(availability_root/'manifest.json')['config']
        if (availability_result['status'] != 'completed_availability_body_diagnostic'
                or availability_result['training_admissible'] is not False
                or sha256(availability_root/'availability-body.npz') != availability_result['arrays_sha256']
                or availability_manifest['flyvis_sha256'] != rr['responses_sha256']
                or availability_manifest['recording_result_sha256'] != read_json(
                    projection/'manifest.json')['config']['recording_result_sha256']):
            raise ValueError('Exact diagnostic group availability body required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'projection_sha256': pr['arrays_sha256'], 'replay_sha256': rr['responses_sha256'],
        'probe_sha256': br['arrays_sha256'], 'joint_checkpoint_sha256': sha256(checkpoint),
        'temporal_checkpoint_sha256': op['checkpoint_sha256'],
        'conditions': ['zero', 'fixed_probe', 'negative_fixed_probe'],
        'body': ('group availability prior receipt' if availability_root else
                 'first-view preceding receipt as historical proxy, not current body'),
        'availability_body_sha256': availability_result['arrays_sha256'] if availability_root else None,
        'task_goal': [0, 0.5], 'dtype': 'float64' if args.double else 'float32',
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'training_admissible': False, 'actor_deployed': False,
              'physical_sample_or_exposure_time_verified': False}
    try:
        with np.load(files[0][0], allow_pickle=False) as p, np.load(files[1][0], allow_pickle=False) as r, np.load(files[2][0], allow_pickle=False) as b:
            retina, body, valid = p['retinal'].copy(), p['body_values'].copy(), p['body_valid'].copy()
            motion, clock = r['motion_activity'].copy(), r['group_source_available_clock'].copy()
            delta = b['probe_delta'].copy()
            np.testing.assert_array_equal(retina, r['retinal_intensity'])
            np.testing.assert_array_equal(valid, r['body_valid'])
            np.testing.assert_array_equal(valid[:, :, 4], b['yaw_valid'])
            np.testing.assert_array_equal(clock, b['group_clock'])
        if availability_root:
            with np.load(availability_root/'availability-body.npz', allow_pickle=False) as group:
                decision_body, decision_valid = group['values'].copy(), group['valid'].copy()
                np.testing.assert_array_equal(clock, group['group_source_available_clock'])
            if decision_body.shape != (len(clock), 21) or decision_valid.shape != decision_body.shape:
                raise ValueError('Exact 21-channel availability body required')
        else:
            decision_body, decision_valid = body[:, 0], valid[:, 0]
        required = [0, 1, 4, 16, 17, 18, 19, 20]
        good = decision_valid[:, required].all(axis=1)
        indices = np.flatnonzero(good)
        if (not len(indices) or not np.array_equal(indices, np.arange(indices[0], len(clock)))
                or not np.isfinite(body).all() or not np.isfinite(decision_body).all()
                or not np.isfinite(motion).all()
                or not np.isfinite(delta).all() or np.any(np.diff(clock) <= 0)):
            raise ValueError('One uninterrupted final segment with required valid body needed')
        start = int(indices[0])
        torch.set_num_threads(1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
            source['checkpoint'], map_location='cpu', weights_only=True)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
        model.eval().requires_grad_(False)
        temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
            model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to(device)
        temporal.load_state_dict(torch.load(files[3][0], map_location='cpu', weights_only=True)['model'], strict=True)
        temporal.eval().requires_grad_(False)
        dtype = torch.float64 if args.double else torch.float32
        if args.double:
            model.double()
            temporal.double()
        if delta.shape != (len(clock), len(model.backbone.vision_ports)):
            raise ValueError('Exact per-group visual port delta required')
        with torch.no_grad():
            scene = model.backbone.scene_pool(torch.as_tensor(np.concatenate(
                (retina[:, :, None], motion), axis=2), dtype=dtype, device=device))
            view_valid = torch.ones((1, 4), dtype=torch.bool, device=device)
            visual = torch.zeros((1, 6), dtype=dtype, device=device)
            visual_valid = torch.zeros((1, 6), dtype=torch.bool, device=device)
            goal = torch.tensor([[0., .5]], dtype=dtype, device=device)
            traces, states, held = None, None, None
            intent_rows, muscle_rows, state_diffs = [], [], []
            for i in range(len(clock)):
                if i == 0:
                    temporal_delta, traces = temporal.step(scene[:1], view_valid, None, None)
                else:
                    steps = integration_intervals(float(clock[i]-clock[i-1]))
                    for j, dt in enumerate(steps):
                        current = scene[i:i+1] if j == len(steps)-1 else scene[i-1:i]
                        temporal_delta, traces = temporal.step(current, view_valid, traces, dt)
                        if i > start:
                            states = model._advance_elapsed_tensor(held, states, dt)
                if i < start:
                    continue
                if i == start:
                    states = model.initial_state(3)
                intent, muscles = model._read_state_tensor(states)
                intent_rows.append(intent.cpu().numpy())
                muscle_rows.append(muscles.cpu().numpy())
                state_diffs.append(torch.max(torch.abs(states[1:] - states[:1]), dim=1).values.cpu().numpy())
                b = torch.as_tensor(decision_body[i:i+1], dtype=dtype, device=device)
                v = torch.as_tensor(decision_valid[i:i+1], dtype=torch.bool, device=device)
                base = model._drive_tensor(visual, visual_valid, b, v, goal, scene[i:i+1])
                base = temporal.modulate(base, model.backbone.vision_ports, temporal_delta)
                d = torch.as_tensor(delta[i:i+1], dtype=dtype, device=device)
                held = torch.cat((base, temporal.modulate(base, model.backbone.vision_ports, d),
                                  temporal.modulate(base, model.backbone.vision_ports, -d)))
        intent, muscles, differences = map(np.stack, (intent_rows, muscle_rows, state_diffs))
        if not all(np.isfinite(x).all() for x in (intent, muscles, differences)):
            raise ValueError('Finite original circuit states and outputs required')
        np.savez_compressed(run/'graph-intervention.npz', group_clock=clock[start:],
            intent=intent, muscles=muscles, state_max_abs_from_neutral=differences)
        result.update(status='completed_offline_signed_graph_intervention', start_group=start,
            computation_dtype='float64' if args.double else 'float32',
            groups=len(clock)-start, initial_graph_state='zero at diagnostic segment start',
            retained_visual_history_groups=start, graph_resets_within_segment=0,
            state_max_abs=differences.max(axis=0).tolist(),
            intent_max_abs=np.max(np.abs(intent[:, 1:]-intent[:, :1]), axis=0).tolist(),
            muscle_max_abs=np.max(np.abs(muscles[:, 1:]-muscles[:, :1]), axis=0).tolist(),
            output_shape=list(intent.shape), optimizer_updates=0,
            actual_current_body_or_wind_verified=False, vehicle_control_or_physiology_verified=False,
            arrays_sha256=sha256(run/'graph-intervention.npz'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
