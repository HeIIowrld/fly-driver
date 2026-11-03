"""Offline original-receipt multirate flight experiment; no actuator or learning."""

import argparse
import json
import time
from bisect import bisect_left, bisect_right
from dataclasses import replace
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
from flybrain_drive.multirate_flight_stream import MultirateFlightStream
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, NavigationBodyObservation
from scripts.audit_causal_copy_body_context import first_observed_packets


def original_view(row, *, episode, camera, contract):
    if row is None:
        return None
    return NavigationBodyObservation(episode, row['packet'], row['receipt'], camera,
        np.asarray(row['values'], np.float32), np.asarray(row['valid'], bool), contract)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'flyvis', 'body-audit', 'neural-audit', 'training', 'opponent', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--cpu-threads', type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument('--scipy-cpu', action='store_true', help='Explicit inference-only SciPy CSR comparison')
    p.add_argument('--body-period-ms', type=float, default=20.)
    p.add_argument('--verify-split-cuda', action='store_true')
    p.add_argument('--body-stream-priority', type=int, choices=(-1, 0), default=0)
    p.add_argument('--compare-cpu-body', action='store_true',
                   help='Compare same recurrent stream on CPU; offline diagnostic only')
    p.add_argument('--body-intervention', choices=('zero_yaw', 'zero_tilt', 'zero_load'),
                   default='zero_yaw', help='Current-body counterfactual; preserve original masks')
    p.add_argument('--probe-graphed-rate', action='store_true')
    a = p.parse_args()
    if a.scipy_cpu and not a.compare_cpu_body:
        raise ValueError('SciPy backend requires the CPU comparison')
    if a.body_stream_priority != 0 and not a.verify_split_cuda:
        raise ValueError('Nondefault priority requires explicit split CUDA verification')
    if not 10 <= a.body_period_ms <= 40:
        raise ValueError('Bounded original-packet subsampling period 10..40ms required')
    root, flyvis, body_audit, neural_audit, training, opponent = map(Path,
        (a.recording, a.flyvis, a.body_audit, a.neural_audit, a.training, a.opponent))
    read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
    sr, vr, br, nr, tr, op = (read(x/'result.json') for x in
                               (root, flyvis, body_audit, neural_audit, training, opponent))
    vm, bm, nm = (read(x/'manifest.json')['config'] for x in (flyvis, body_audit, neural_audit))
    checks = ((root/'online-body-shadow.json', sr['shadow_sha256']),
              (root/'native-copy-windows.json', sr['native_copy_sha256']),
              (flyvis/'responses.npz', vr['responses_sha256']),
              (opponent/'temporal-boundary.pt', op['checkpoint_sha256']))
    if (sr['status'] != 'completed' or sr['training_admissible'] is not False
            or br['status'] != 'completed_online_body_shadow_audit'
            or nr['status'] != 'completed_live_neural_shadow_audit'
            or vm['recording_result_sha256'] != sha256(root/'result.json')
            or vm['online_shadow_sha256'] != sr['shadow_sha256']
            or bm['recording_sha256'] != sha256(root/'result.json')
            or nm['recording_result_sha256'] != sha256(root/'result.json')
            or vr['clock_source'] != 'original host visual snapshot preceding actual visual completion'
            or op['source_checkpoint_sha256'] != sha256(Path(tr['checkpoint']))
            or any(sha256(path) != expected for path, expected in checks)):
        raise ValueError('Exact audited live source, FlyVis timing and model lineage required')
    events, rows = read(checks[0][0]), read(checks[1][0])
    packets = first_observed_packets(rows)
    receipts = [packet['first_end'] for packet in packets]
    with np.load(checks[2][0], allow_pickle=False) as data:
        retina, motion, clocks = (data[key].copy() for key in
            ('retinal_intensity', 'motion_activity', 'group_source_available_clock'))
    np.testing.assert_array_equal(clocks, [event['visual_snapshot'] for event in events])
    valid_indices = [i for i, e in enumerate(events) if e['neural']['intent'] is not None]
    if (not valid_indices or valid_indices != list(range(valid_indices[0], valid_indices[-1]+1))
            or any(events[i]['context_ready'] < events[i]['neural']['vision_finished']
                   for i in valid_indices)):
        raise ValueError('One original continuous body-valid visual segment required')
    first, last = valid_indices[0], valid_indices[-1]
    plan = read(training/'manifest.json')['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'multirate_source_sha256': sha256('src/flybrain_drive/multirate_flight_stream.py'),
        'recording_sha256': sha256(root/'result.json'), 'flyvis_sha256': vr['responses_sha256'],
        'body_audit_sha256': sha256(body_audit/'result.json'),
        'neural_audit_sha256': sha256(neural_audit/'result.json'),
        'flight_checkpoint_sha256': op['source_checkpoint_sha256'],
        'opponent_checkpoint_sha256': op['checkpoint_sha256'], 'body_period_ms': a.body_period_ms,
        'verify_split_cuda': a.verify_split_cuda,
        'body_stream_priority': a.body_stream_priority,
        'compare_cpu_body': a.compare_cpu_body, 'cpu_threads': a.cpu_threads,
        'cpu_backend': 'scipy_csr_cpu' if a.scipy_cpu else 'cached_csr_cpu',
        'live_cuda_source_sha256': sha256('src/flybrain_drive/multirate_live_shadow.py')
        if a.verify_split_cuda else None,
        'probe_graphed_rate': a.probe_graphed_rate,
        'graphed_rate_source_sha256': sha256('src/flybrain_drive/graphed_body_rate.py')
        if a.probe_graphed_rate else None,
        'scope': 'ideal offline scheduling, host receipt time not physical time',
        'body_intervention': a.body_intervention,
        'yaw_intervention': 'current body yaw set to zero, validity preserved; explicit counterfactual'
        if a.body_intervention == 'zero_yaw' else None,
        'training_admissible': False, 'actor_input': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'training_admissible': False, 'actor_deployed': False}
    try:
        torch.set_num_threads(a.cpu_threads)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if (a.verify_split_cuda or a.probe_graphed_rate) and device != 'cuda':
            raise RuntimeError('Split-stream verification requires the original GPU backend')
        model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
            source['checkpoint'], map_location='cpu', weights_only=True)).to(device)
        model.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
        model.eval().requires_grad_(False)
        temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
            model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to(device)
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt', map_location='cpu', weights_only=True)['model'], strict=True)
        joint = ExperimentalVisualBodyAirflow(model).to(device).eval().requires_grad_(False)
        if a.probe_graphed_rate:
            from flybrain_drive.graphed_body_rate import GraphedHeldBodyRate
            from flybrain_drive.flyvis_camera import integration_intervals
            rate_graph = GraphedHeldBodyRate(joint)
        else:
            rate_graph = None
        boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device=device)
        streams = [MultirateFlightStream(PerViewTemporalContextOpponentFlightStream(
            joint, temporal, boundary, optimize_held=True)) for _ in range(3)]
        split = (MultirateFlightStream(PerViewTemporalContextOpponentFlightStream(
            joint, temporal, boundary, optimize_held=True)) if a.verify_split_cuda else None)
        split_cuda = (torch.cuda.Stream(device=device, priority=a.body_stream_priority)
                      if split is not None else None)
        if split_cuda is not None and split_cuda.priority != a.body_stream_priority:
            raise ValueError('Runtime did not honor the requested priority')
        cpu_stream, cpu_kwargs = None, None
        if a.compare_cpu_body:
            if device != 'cuda':
                raise ValueError('Cross-device comparison requires the original GPU backend')
            cpu_model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
                source['checkpoint'], map_location='cpu', weights_only=True)).to('cpu')
            cpu_model.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu',
                                                weights_only=True)['model'], strict=True)
            cpu_model.backbone.circuit.sparse_backend = 'scipy_csr_cpu' if a.scipy_cpu else 'cached_csr_cpu'
            cpu_model.eval().requires_grad_(False)
            cpu_temporal = OpponentTemporalMotionBoundary(cpu_model.backbone.scene_command.mean,
                cpu_model.backbone.scene_command.scale, len(cpu_model.backbone.vision_ports))
            cpu_temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',
                                                   map_location='cpu', weights_only=True)['model'],
                                         strict=True)
            cpu_joint = ExperimentalVisualBodyAirflow(cpu_model).eval().requires_grad_(False)
            cpu_boundary = PerViewVisualBodyContextBoundary(len(cpu_model.backbone.vision_ports),
                                                            device='cpu')
            cpu_stream = MultirateFlightStream(PerViewTemporalContextOpponentFlightStream(
                cpu_joint, cpu_temporal, cpu_boundary, optimize_held=True))
            cpu_kwargs = dict(goals=torch.tensor([[0., .5]]), visual=torch.zeros(1, 6),
                              visual_valid=torch.zeros(1, 6, dtype=torch.bool),
                              scene_valid=torch.ones(1, 4, dtype=torch.bool))
        # A: body-rate; B: identical timing with explicit current yaw=0 lesion;
        # C: original image-rate body updates. All keep same FlyVis features.
        identity = read('configs/ac-virtual-body-v1.json')['native_identity']
        encoders = [NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                           native_identity=identity) for _ in range(2)]
        episode = sr['request_id']
        contract = sha256('configs/ac-virtual-navigation-body-v2.json')
        visual = torch.zeros(1, 6, device=device)
        visual_valid = torch.zeros(1, 6, dtype=torch.bool, device=device)
        scene_valid = torch.ones(1, 4, dtype=torch.bool, device=device)
        goals = torch.tensor([[0., .5]], device=device)
        kwargs = dict(goals=goals, visual=visual, visual_valid=visual_valid, scene_valid=scene_valid)
        with torch.no_grad():
            scenes = model.backbone.scene_pool(torch.as_tensor(np.concatenate(
                (retina[:, :, None], motion), axis=2), dtype=torch.float32, device=device))
            if split is not None:
                # All model buffers and scenes are ready before handing them to a new stream.
                torch.cuda.synchronize()
            records, frame_rows, processing, last_tick = [], [], [], None
            split_diffs = []
            cpu_diffs, cpu_processing = [], []
            rate_diffs, rate_reference_ms, rate_graph_ms = [], [], []
            intervention_indices = {'zero_yaw': (4,), 'zero_tilt': (0, 1),
                                    'zero_load': (8, 9, 10, 11)}[a.body_intervention]
            observed_intervention_values = []
            period = a.body_period_ms/1000.

            def update_fast(when, packet, scene=None, context=None, kind='body'):
                nonlocal last_tick
                body = encoders[0].encode(packet['sample'], episode_id=episode,
                                           decision_monotonic_seconds=when)
                damaged = body.values.copy()
                observed_intervention_values.extend(abs(float(body.values[i]))
                    for i in intervention_indices if body.valid[i])
                damaged[list(intervention_indices)] = 0.
                counterfactual = replace(body, values=damaged)
                started = time.perf_counter()
                intent, _, info = streams[0].update(when=when, body=body, new_scene=scene,
                    new_visual_context=context, **kwargs)
                intent = intent.cpu().numpy()[0]
                processing.append(1000*(time.perf_counter()-started))
                if rate_graph is not None:
                    elapsed = .0037 if last_tick is None else when-last_tick
                    intervals = integration_intervals(elapsed)
                    original_state = streams[0].stream.state
                    original_drive = streams[0].stream.previous_drive
                    torch.cuda.synchronize()
                    started_probe = time.perf_counter()
                    reference_state = original_state
                    for dt in intervals:
                        reference_state = joint.circuit._advance_elapsed_tensor(
                            original_drive, reference_state, dt)
                    torch.cuda.synchronize()
                    rate_reference_ms.append(1000*(time.perf_counter()-started_probe))
                    started_probe = time.perf_counter()
                    captured_state = rate_graph.advance(original_drive, original_state, intervals)
                    torch.cuda.synchronize()
                    rate_graph_ms.append(1000*(time.perf_counter()-started_probe))
                    rate_diffs.append(torch.max(torch.abs(reference_state-captured_state)).item())
                lesion, _, _ = streams[1].update(when=when, body=counterfactual, new_scene=scene,
                    new_visual_context=context, **kwargs)
                lesion = lesion.cpu().numpy()[0]
                if split is not None:
                    with torch.cuda.stream(split_cuda):
                        split_intent, _, _ = split.update(when=when, body=body, new_scene=scene,
                            new_visual_context=context, **kwargs)
                        split_intent.cpu()
                        split_cuda.synchronize()
                    normal, other = streams[0].stream, split.stream
                    values = [torch.max(torch.abs(normal.state-other.state)).item(),
                              torch.max(torch.abs(normal.previous_drive-other.previous_drive)).item()]
                    values.extend(torch.max(torch.abs(x-y)).item() for x, y in
                                  zip(normal.traces, other.traces, strict=True))
                    values.append(float(np.max(np.abs(intent-split_intent.cpu().numpy()[0]))))
                    split_diffs.append(values)
                if cpu_stream is not None:
                    cpu_scene = None if scene is None else scene.detach().cpu()
                    cpu_started = time.perf_counter()
                    cpu_intent, _, _ = cpu_stream.update(when=when, body=body,
                        new_scene=cpu_scene, new_visual_context=context, **cpu_kwargs)
                    cpu_processing.append(1000*(time.perf_counter()-cpu_started))
                    normal, other = streams[0].stream, cpu_stream.stream
                    differences = [torch.max(torch.abs(normal.state.cpu()-other.state)).item(),
                                   torch.max(torch.abs(normal.previous_drive.cpu()-
                                                       other.previous_drive)).item()]
                    differences.extend(torch.max(torch.abs(x.cpu()-y)).item() for x, y in
                                       zip(normal.traces, other.traces, strict=True))
                    differences.append(float(np.max(np.abs(intent-cpu_intent.numpy()[0]))))
                    cpu_diffs.append(differences)
                row = {'kind': kind, 'time': when, 'packet': body.packet_id,
                    'held_age_ms': None if info['previous_held_body_age_at_decision'] is None else
                        1000*info['previous_held_body_age_at_decision'],
                    'intent': intent.tolist()}
                suffix = 'yaw_zero' if a.body_intervention == 'zero_yaw' else 'counterfactual'
                row[suffix+'_intent'] = lesion.tolist()
                state_key = ('state_yaw_lesion_max_abs' if suffix == 'yaw_zero'
                             else 'state_counterfactual_max_abs')
                row[state_key] = float(torch.max(torch.abs(
                    streams[0].stream.state-streams[1].stream.state)).item())
                records.append(row)
                last_tick = when

            for i in range(first, last+1):
                event = events[i]
                when = event['context_ready']
                recorded_index = bisect_right(receipts, event['decision'])-1
                if (recorded_index <= 0
                        or packets[recorded_index]['packet'] != event['decision_body']['packet']):
                    raise ValueError('Actual original camera-context body differs')
                # Hypothetical independent consumer may use any original packet
                # already received by scene availability, never one after it.
                index = bisect_right(receipts, when)-1
                body = encoders[1].encode(packets[index]['sample'], episode_id=episode,
                                           decision_monotonic_seconds=when)
                view_bodies = tuple(original_view(row, episode=episode, camera=event['camera_receipt'],
                    contract=contract) for row in event['view_body'])
                frame_context = per_view_temporal_body_context(view_bodies, body,
                    render_start_lowers=tuple(event['render_start_lower']),
                    camera_receipt_time=event['camera_receipt'], decision_time=when)
                update_fast(when, packets[index], scenes[i:i+1], frame_context, 'frame')
                reference, _, ref_info = streams[2].update(when=when, body=body,
                    new_scene=scenes[i:i+1], new_visual_context=frame_context, **kwargs)
                frame_rows.append({'index': i, 'time': when, 'fast_intent': records[-1]['intent'],
                    'frame_only_intent': reference.cpu().numpy()[0].tolist(),
                    'fast_held_age_ms': records[-1]['held_age_ms'],
                    'frame_only_held_age_ms': None if ref_info['previous_held_body_age_at_decision'] is None
                        else 1000*ref_info['previous_held_body_age_at_decision']})
                if i == last:
                    break
                end = events[i+1]['context_ready']
                for pi in range(bisect_right(receipts, when), bisect_left(receipts, end)):
                    packet = packets[pi]
                    stamp = packet['first_end']
                    if stamp-last_tick >= period:
                        update_fast(stamp, packet)
        with (run/'events.json').open('x', encoding='utf8') as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
        write_json(run/'frame-comparison.json', frame_rows)
        ages = [r['held_age_ms'] for r in records if r['held_age_ms'] is not None]
        frame_ages = [r['frame_only_held_age_ms'] for r in frame_rows if r['frame_only_held_age_ms'] is not None]
        suffix = 'yaw_zero' if a.body_intervention == 'zero_yaw' else 'counterfactual'
        state_key = ('state_yaw_lesion_max_abs' if suffix == 'yaw_zero'
                     else 'state_counterfactual_max_abs')
        delta = np.asarray([np.asarray(r['intent'])-r[suffix+'_intent'] for r in records])
        frame_delta = np.asarray([np.asarray(r['fast_intent'])-r['frame_only_intent'] for r in frame_rows])
        if split is not None and (len(split_diffs) != len(records)
                or not np.isfinite(split_diffs).all() or np.max(split_diffs) > 2e-6):
            raise ValueError('Split CUDA state/drive/traces/intent diverged from reference')
        if rate_graph is not None and (len(rate_diffs) != len(records)
                or not np.isfinite(rate_diffs).all() or max(rate_diffs) > 2e-6):
            raise ValueError('Captured held rate differs from original explicit intervals')
        if cpu_stream is not None and (len(cpu_diffs) != len(records)
                or not np.isfinite(cpu_diffs).all() or np.max(cpu_diffs) > 2e-6):
            raise ValueError('CPU state/drive/traces/intent diverged from GPU reference')
        result.update(status='completed_offline_multirate_flight_replay', frames=len(frame_rows),
            body_updates=sum(r['kind'] == 'body' for r in records), total_updates=len(records),
            fast_held_age_ms_p50_p95_max=np.percentile(ages, [50, 95, 100]).tolist(),
            fast_held_age_over_50ms=sum(x > 50 for x in ages),
            frame_only_held_age_ms_p50_p95_max=np.percentile(frame_ages, [50, 95, 100]).tolist(),
            frame_only_held_age_over_50ms=sum(x > 50 for x in frame_ages),
            multirate_vs_frame_only_intent_max_abs=np.max(np.abs(frame_delta), axis=0).tolist(),
            fast_stream_only_ms_p50_p95_max=np.percentile(processing, [50, 95, 100]).tolist(),
            timings_exclude_visual_body_decode_other_conditions_and_live_scheduling=True,
            intent_age_evaluated_at_model_event_not_real_output=True,
            source_physical_time_or_exposure_verified=False,
            previous_visual_held_between_updates=True, original_episode_resets=0,
            optimizer_updates=0, events_sha256=sha256(run/'events.json'),
            frame_comparison_sha256=sha256(run/'frame-comparison.json'))
        result.update(body_intervention=a.body_intervention,
            intervention_indices=list(intervention_indices),
            intervention_valid_values=len(observed_intervention_values),
            intervention_original_max_abs=max(observed_intervention_values, default=0.),
            counterfactual_intent_max_abs=np.max(np.abs(delta), axis=0).tolist(),
            counterfactual_state_max_abs=max(r[state_key] for r in records),
            intervention_is_causal_physiology_or_driving_proof=False)
        if a.body_intervention == 'zero_yaw':
            result.update(yaw_zero_intent_max_abs=result['counterfactual_intent_max_abs'],
                          yaw_zero_state_max_abs=result['counterfactual_state_max_abs'])
        if split is not None:
            result.update(split_cuda_compared_events=len(split_diffs),
                split_cuda_max_state_drive_trace_intent_abs=float(np.max(split_diffs)),
                split_cuda_identical_all=bool(np.max(split_diffs) == 0),
                split_cuda_tolerance=2e-6,
                split_cuda_concurrent_vision_tested=False)
        if cpu_stream is not None:
            write_json(run/'cpu-timings.json', cpu_processing)
            result.update(cpu_compared_events=len(cpu_diffs),
                cpu_timing_sha256=sha256(run/'cpu-timings.json'),
                cpu_max_state_drive_trace_intent_abs=float(np.max(cpu_diffs)),
                cpu_stream_only_ms_p50_p95_max=np.percentile(cpu_processing, [50, 95, 100]).tolist(),
                cpu_timing_excludes_scene_transfer_and_concurrent_vision=True,
                cpu_live_scheduler_verified=False)
        if rate_graph is not None:
            result.update(graphed_rate_probed_events=len(rate_diffs),
                graphed_rate_max_state_abs=float(max(rate_diffs)),
                graphed_rate_identical_all=bool(max(rate_diffs) == 0),
                graphed_rate_reference_ms_p50_p95_max=np.percentile(rate_reference_ms, [50, 95, 100]).tolist(),
                graphed_rate_captured_ms_p50_p95_max=np.percentile(rate_graph_ms, [50, 95, 100]).tolist(),
                graphed_rate_full_recurrent_stream_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
