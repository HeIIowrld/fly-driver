"""Offline full FlyVis-output + original online body-context flight replay; no control."""

import argparse
import json
import time
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
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def observation(row, *, episode, contract, evaluation):
    if row is None:
        return None
    return NavigationBodyObservation(episode, row['packet'], row['receipt'], evaluation,
        np.asarray(row['values'], dtype=np.float32), np.asarray(row['valid'], dtype=bool), contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'flyvis', 'training', 'opponent', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--check-optimized', action='store_true')
    args = parser.parse_args()
    root, flyvis, training, opponent = map(Path,
        (args.recording, args.flyvis, args.training, args.opponent))
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    sr, fr, tr, op = (read(x/'result.json') for x in (root, flyvis, training, opponent))
    fm = read(flyvis/'manifest.json')['config']
    if (sr['status'] != 'completed' or sr['training_admissible'] is not False
            or sr['shadow_sha256'] != sha256(root/'online-body-shadow.json')
            or fr['status'] != 'completed_diagnostic_flyvis_replay'
            or fr['responses_sha256'] != sha256(flyvis/'responses.npz')
            or fm['online_shadow_sha256'] != sr['shadow_sha256']
            or fm['recording_result_sha256'] != sha256(root/'result.json')
            or op['checkpoint_sha256'] != sha256(opponent/'temporal-boundary.pt')
            or op['source_checkpoint_sha256'] != sha256(Path(tr['checkpoint']))):
        raise ValueError('Exact recorded shadow, FlyVis replay and model lineage required')
    events = read(root/'online-body-shadow.json')
    with np.load(flyvis/'responses.npz', allow_pickle=False) as data:
        retina, motion, times = (data[k].copy() for k in
            ('retinal_intensity', 'motion_activity', 'group_source_available_clock'))
    np.testing.assert_array_equal(times, np.asarray([event['decision'] for event in events]))
    required = [0, 1, 4, 16, 17, 18, 19, 20]
    good = np.asarray([event['status'] == 'context_built' and all(
        event['decision_body']['valid'][k] for k in required) for event in events])
    indices = np.flatnonzero(good)
    if not len(indices) or not np.array_equal(indices, np.arange(indices[0], indices[-1]+1)):
        raise ValueError('One contiguous valid body tail required')
    start, end = int(indices[0]), int(indices[-1])+1
    plan = read(training/'manifest.json')['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'recording_sha256': sha256(root/'result.json'), 'online_shadow_sha256': sr['shadow_sha256'],
        'flyvis_sha256': fr['responses_sha256'], 'joint_checkpoint_sha256': op['source_checkpoint_sha256'],
        'opponent_checkpoint_sha256': op['checkpoint_sha256'], 'segment': [start, end],
        'probe_weights': '0.01/256', 'check_optimized': args.check_optimized,
        'training_admissible': False, 'actor_input': False})
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
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt', map_location='cpu', weights_only=True)['model'], strict=True)
        joint = ExperimentalVisualBodyAirflow(model).to(device).eval().requires_grad_(False)
        boundaries = [PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device=device)
                      for _ in range(2)]
        with torch.no_grad():
            boundaries[1].projection.weight.fill_(.01/256)
        streams = [PerViewTemporalContextOpponentFlightStream(joint, temporal, b) for b in boundaries]
        optimized = ([PerViewTemporalContextOpponentFlightStream(joint, temporal, b,
                      optimize_held=True) for b in boundaries] if args.check_optimized else [])
        contract = sha256('configs/ac-virtual-navigation-body-v2.json')
        with torch.no_grad():
            scene = model.backbone.scene_pool(torch.as_tensor(np.concatenate(
                (retina[:, :, None], motion), axis=2), dtype=torch.float32, device=device))
            visual = torch.zeros(1, 6, device=device)
            visual_valid = torch.zeros(1, 6, dtype=torch.bool, device=device)
            scene_valid = torch.ones(1, 4, dtype=torch.bool, device=device)
            goals = torch.tensor([[0., .5]], device=device)
            output, elapsed, optimized_elapsed, differences = [], [], [], []
            for i in range(start, end):
                event = events[i]
                current = observation(event['decision_body'], episode=sr['request_id'],
                                      contract=contract, evaluation=event['decision'])
                views = tuple(observation(row, episode=sr['request_id'], contract=contract,
                    evaluation=event['camera_receipt']) for row in event['view_body'])
                context = per_view_temporal_body_context(views, current,
                    render_start_lowers=tuple(event['render_start_lower']),
                    camera_receipt_time=event['camera_receipt'], decision_time=event['decision'])
                start_cpu = time.perf_counter()
                row = []
                for stream in streams:
                    intent, _ = stream.decision(decision_time=event['decision'], visual=visual,
                        visual_valid=visual_valid, body_context=context, goals=goals,
                        scene_features=scene[i:i+1], scene_valid=scene_valid,
                        airflow_observation=missing_airflow(event['decision']))
                    row.append(intent.cpu().numpy()[0])
                elapsed.append(1000*(time.perf_counter()-start_cpu))
                output.append(row)
                if optimized:
                    start_cpu = time.perf_counter()
                    for stream in optimized:
                        intent, _ = stream.decision(decision_time=event['decision'], visual=visual,
                            visual_valid=visual_valid, body_context=context, goals=goals,
                            scene_features=scene[i:i+1], scene_valid=scene_valid,
                            airflow_observation=missing_airflow(event['decision']))
                        intent.cpu()
                    optimized_elapsed.append(1000*(time.perf_counter()-start_cpu))
                    diff = []
                    for base, fast in zip(streams, optimized, strict=True):
                        diff.extend([torch.max(torch.abs(base.state-fast.state)).item(),
                            torch.max(torch.abs(base.previous_drive-fast.previous_drive)).item()])
                        diff.extend(torch.max(torch.abs(x-y)).item()
                                    for x, y in zip(base.traces, fast.traces, strict=True))
                    differences.append(diff)
        output = np.asarray(output)
        if not np.isfinite(output).all():
            raise ValueError('Finite original flight responses required')
        if optimized and (not np.isfinite(differences).all() or np.max(differences) > 2e-6):
            raise ValueError('Optimized trace/state/drive differs from reference beyond 2e-6')
        np.savez_compressed(run/'outputs.npz', intent=output, decision_clock=times[start:end])
        result.update(status='completed_offline_online_context_flight_replay', groups=len(output),
            start_group=start, end_group_exclusive=end, visual_history_groups=start,
            initial_flight_state_reset=True, original_full_flight_history_replayed=False,
            probe_intent_max_abs=np.max(np.abs(output[:, 1]-output[:, 0]), axis=0).tolist(),
            two_flight_stream_processing_ms_p50_p95_max=np.percentile(elapsed, [50, 95, 100]).tolist(),
            timing_scope='offline two flight streams only; excludes FlyVis, images and body build',
            arrays_sha256=sha256(run/'outputs.npz'), optimizer_updates=0,
            physical_sample_exposure_or_live_neural_decision_verified=False)
        if optimized:
            result.update(optimized_conditions=2, optimized_max_state_drive_or_trace_abs=float(
                np.max(differences)), optimized_identical_all=bool(np.max(differences) == 0),
                optimized_two_stream_processing_ms_p50_p95_max=np.percentile(
                    optimized_elapsed, [50, 95, 100]).tolist(),
                optimized_error_tolerance=2e-6)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
