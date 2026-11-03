"""Offline host-timed visual/body interaction probe; no actor, learning, or control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('projection', 'replay', 'training', 'opponent', 'output'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    projection, replay, training, opponent = map(Path, (a.projection, a.replay, a.training, a.opponent))
    read = lambda p: json.loads(p.read_text(encoding='utf8'))
    pr, rr, tr, op = (read(p/'result.json') for p in (projection, replay, training, opponent))
    if (pr['arrays_sha256'] != sha256(projection/'retina-body-diagnostic.npz')
            or rr['responses_sha256'] != sha256(replay/'responses.npz')
            or op['checkpoint_sha256'] != sha256(opponent/'temporal-boundary.pt')
            or rr['status'] != 'completed_diagnostic_flyvis_replay'
            or rr['training_admissible'] is not False):
        raise ValueError('Verified original diagnostic and checkpoints required')
    plan = read(training/'manifest.json')['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    checkpoint = Path(tr['checkpoint'])
    if sha256(checkpoint) != op['source_checkpoint_sha256']:
        raise ValueError('Temporal checkpoint must match its trained parent')
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'projection_sha256': pr['arrays_sha256'], 'flyvis_sha256': rr['responses_sha256'],
        'joint_checkpoint_sha256': sha256(checkpoint), 'temporal_checkpoint_sha256': op['checkpoint_sha256'],
        'probe': 'all projection weights 0.01/256; fixed before inspecting response',
        'training_admissible': False, 'actor_input': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'training_admissible': False, 'actor_deployed': False,
              'physical_sample_or_exposure_time_verified': False}
    try:
        with np.load(projection/'retina-body-diagnostic.npz', allow_pickle=False) as p, np.load(replay/'responses.npz', allow_pickle=False) as r:
            retina, body, valid = p['retinal'].copy(), p['body_values'].copy(), p['body_valid'].copy()
            motion, clocks = r['motion_activity'].copy(), r['group_source_available_clock'].copy()
            np.testing.assert_array_equal(retina, r['retinal_intensity'])
            np.testing.assert_array_equal(valid, r['body_valid'])
        if (body.shape != (len(clocks), 4, 21) or valid.shape != body.shape
                or motion.shape != (len(clocks), 4, 8, 721) or not np.isfinite(body).all()
                or not np.isfinite(motion).all() or not np.all(np.diff(clocks) > 0)):
            raise ValueError('Complete finite original ordered inputs required')
        torch.set_num_threads(1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(source['checkpoint'], map_location='cpu', weights_only=True)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
        model.eval().requires_grad_(False)
        temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean, model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to(device)
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt', map_location='cpu', weights_only=True)['model'], strict=True)
        temporal.eval().requires_grad_(False)
        neutral = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device=device)
        probe = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device=device)
        with torch.no_grad():
            probe.projection.weight.fill_(.01/256)
            scene = model.backbone.scene_pool(torch.as_tensor(np.concatenate((retina[:, :, None], motion), axis=2), dtype=torch.float32, device=device))
            all_views = torch.ones((1, 4), dtype=torch.bool, device=device)
            state, rows, zeros, ticks = None, [], [], 0
            for i in range(len(clocks)):
                if i == 0:
                    _, state = temporal.step(scene[i:i+1], all_views, None, None)
                else:
                    intervals = integration_intervals(float(clocks[i]-clocks[i-1]))
                    for j, dt in enumerate(intervals):
                        current = scene[i:i+1] if j == len(intervals)-1 else scene[i-1:i]
                        _, state = temporal.step(current, all_views, state, dt)
                        ticks += 1
                b = torch.as_tensor(body[i:i+1], dtype=torch.float32, device=device)
                v = torch.as_tensor(valid[i:i+1], dtype=torch.bool, device=device)
                rows.append(probe(state, b, v).cpu().numpy()[0])
                zeros.append(neutral(state, b, v).cpu().numpy()[0])
        rows, zeros = np.stack(rows), np.stack(zeros)
        missing = ~valid[:, :, 4].any(axis=1)
        if np.any(zeros != 0) or np.any(rows[missing] != 0) or not np.isfinite(rows).all():
            raise ValueError('Neutral and completely missing yaw must produce zero')
        np.savez_compressed(run/'port-probe.npz', probe_delta=rows, neutral_delta=zeros,
                            yaw=body[:, :, 4], yaw_valid=valid[:, :, 4], group_clock=clocks)
        result.update(status='completed_offline_visual_body_port_probe', groups=len(clocks), ticks=ticks,
            missing_yaw_groups=int(missing.sum()), groups_with_yaw=int((~missing).sum()),
            neutral_max_abs=float(np.max(np.abs(zeros))), probe_max_abs=float(np.max(np.abs(rows))),
            probe_nonzero_groups=int(np.count_nonzero(np.any(rows != 0, axis=1))),
            original_group_resets=0, optimizer_updates=0, fixed_probe_learned=False,
            graph_response_or_vehicle_control_tested=False, arrays_sha256=sha256(run/'port-probe.npz'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
