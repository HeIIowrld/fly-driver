"""Replay observed diagnostic segment through neutral/probe four-view stream; no control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.flight_visual_body_temporal_context import (
    PerViewTemporalContextOpponentFlightStream, PerViewVisualBodyContextBoundary,
)
from flybrain_drive.graph import sha256
from flybrain_drive.relative_airflow_sensor import missing_airflow
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, NavigationBodyObservation
from scripts.audit_causal_copy_body_context import first_observed_packets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'body', 'availability', 'projection', 'replay', 'training',
                 'opponent', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    root, body, avail, proj, replay, training, opponent = map(Path,
        (a.recording, a.body, a.availability, a.projection, a.replay, a.training, a.opponent))
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    sr, br, ar, pr, rr, tr, op = (read(x/'result.json') for x in
                                     (root, body, avail, proj, replay, training, opponent))
    am = read(avail/'manifest.json')['config']
    checks = ((root/'native-copy-windows.json', sr['native_copy_sha256']),
              (body/'view-receipts.json', br['view_receipts_sha256']),
              (avail/'receipts.json', ar['receipts_sha256']),
              (avail/'availability-body.npz', ar['arrays_sha256']),
              (proj/'retina-body-diagnostic.npz', pr['arrays_sha256']),
              (replay/'responses.npz', rr['responses_sha256']),
              (opponent/'temporal-boundary.pt', op['checkpoint_sha256']))
    plan = read(training/'manifest.json')['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    if (any(sha256(path) != expected for path, expected in checks)
            or am['recording_result_sha256'] != sha256(root/'result.json')
            or am['flyvis_sha256'] != rr['responses_sha256']
            or sha256(Path(tr['checkpoint'])) != op['source_checkpoint_sha256']
            or ar['training_admissible'] is not False):
        raise ValueError('Matching unchanged diagnostic source and checkpoints required')
    per_view, current = read(body/'view-receipts.json'), read(avail/'receipts.json')
    packet = {x['packet']: x for x in first_observed_packets(read(root/'native-copy-windows.json'))}
    with np.load(proj/'retina-body-diagnostic.npz') as pz, np.load(replay/'responses.npz') as rz, np.load(avail/'availability-body.npz') as az:
        retina, motion = pz['retinal'].copy(), rz['motion_activity'].copy()
        clock, bodies, masks = az['group_source_available_clock'].copy(), az['values'].copy(), az['valid'].copy()
        np.testing.assert_array_equal(clock, rz['group_source_available_clock'])
        np.testing.assert_array_equal(retina, rz['retinal_intensity'])
    required = [0, 1, 4, 16, 17, 18, 19, 20]
    indices = np.flatnonzero(masks[:, required].all(axis=1))
    if not len(indices) or not np.array_equal(indices, np.arange(indices[0], len(clock))):
        raise ValueError('Contiguous fully valid body tail required')
    start = int(indices[0])
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'source_sha256': sha256(root/'result.json'), 'availability_sha256': ar['arrays_sha256'],
        'flyvis_sha256': rr['responses_sha256'], 'opponent_sha256': op['checkpoint_sha256'],
        'checkpoint_sha256': op['source_checkpoint_sha256'], 'segment_start': start,
        'probe_weights': '0.01/256', 'actor_input': False, 'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'training_admissible': False, 'actor_deployed': False}
    try:
        torch.set_num_threads(1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
            source['checkpoint'], map_location='cpu', weights_only=True)).to(device)
        model.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
        model.eval().requires_grad_(False)
        temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
            model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to(device)
        temporal.load_state_dict(torch.load(checks[-1][0], map_location='cpu', weights_only=True)['model'], strict=True)
        joint = ExperimentalVisualBodyAirflow(model).to(device).eval().requires_grad_(False)
        boundaries = [PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device=device)
                      for _ in range(2)]
        with torch.no_grad():
            boundaries[1].projection.weight.fill_(.01/256)
        streams = [PerViewTemporalContextOpponentFlightStream(joint, temporal, b) for b in boundaries]
        identity = read(Path('configs/ac-virtual-body-v1.json'))['native_identity']
        encoders = [NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                           native_identity=identity) for _ in range(4)]
        with torch.no_grad():
            scene = model.backbone.scene_pool(torch.as_tensor(np.concatenate(
                (retina[:, :, None], motion), axis=2), dtype=torch.float32, device=device))
            visual = torch.zeros(1, 6, device=device)
            visual_valid = torch.zeros(1, 6, dtype=torch.bool, device=device)
            scene_valid = torch.ones(1, 4, dtype=torch.bool, device=device)
            goals = torch.tensor([[0., .5]], device=device)
            outputs, missing = [], 0
            for i in range(start, len(clock)):
                record = current[i]
                when = float(record['availability_host_bounds'][1])
                now = NavigationBodyObservation('diagnostic', record['packet_id'],
                    record['first_receipt'], when, bodies[i], masks[i], encoders[0].contract_sha256)
                contexts, bounds = [], []
                for j in range(4):
                    v = per_view[4*i+j]
                    if v['status'] != 'original_first_packet_receipt':
                        raise ValueError('Expected valid original view receipt on this segment')
                    bounds.append(float(v['render_host_bounds'][0]))
                    observation = encoders[j].encode(packet[v['packet_id']]['sample'],
                        episode_id='diagnostic', decision_monotonic_seconds=when)
                    contexts.append(observation)
                    missing += not bool(observation.valid[4])
                ctx = per_view_temporal_body_context(tuple(contexts), now,
                    render_start_lowers=tuple(bounds), camera_receipt_time=when, decision_time=when)
                row = []
                for stream in streams:
                    intent, _ = stream.decision(decision_time=when, visual=visual,
                        visual_valid=visual_valid, body_context=ctx, goals=goals,
                        scene_features=scene[i:i+1], scene_valid=scene_valid,
                        airflow_observation=missing_airflow(when))
                    row.append(intent.cpu().numpy()[0])
                outputs.append(row)
        outputs = np.asarray(outputs)
        if not np.isfinite(outputs).all():
            raise ValueError('Finite stream outputs required')
        np.savez_compressed(run/'outputs.npz', intent=outputs, group_clock=clock[start:])
        result.update(status='completed_diagnostic_stream_replay', groups=len(outputs),
            initial_segment_state_reset=True, original_full_history_replayed=False,
            view_yaw_masked_at_availability=int(missing),
            probe_intent_max_abs=np.max(np.abs(outputs[:, 1]-outputs[:, 0]), axis=0).tolist(),
            arrays_sha256=sha256(run/'outputs.npz'), optimizer_updates=0,
            physical_sample_exposure_or_actual_actor_decision_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
