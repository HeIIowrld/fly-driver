"""Audit one actual physical yaw trial from complete source pixels/native bytes/times."""

import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_vision import validate_frame_aid_state
from flybrain_drive.ac_diagnostic_admission import verified_diagnostic_transport
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.ac_transport import decode_telemetry
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_counterfactual import MODES, FlightCounterfactualReplay
from flybrain_drive.flight_fast_replay import FastOriginalReplay
from flybrain_drive.flight_fast_stream import FastFlightStream, verified_fast_plan
from flybrain_drive.flight_perturbation import validate_perturbation_plan, yaw_metrics
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.flight_yaw_trial import apply_yaw_mode
from flybrain_drive.flyvis_frozen_tail import install_frozen_visual_tail
from flybrain_drive.flyvis_graphed_tail import install_graphed_visual_tail
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.audit_ac_timed_steering import verified_timed_steering
from scripts.record_ac_engine_ai_probe import verified_shadow
from scripts.verify_flight_yaw_trial import verified_yaw_trial


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def native(row):
    raw = base64.b64decode(row['prefix_base64'], validate=True)
    assert hashlib.sha256(raw).hexdigest() == row['prefix_sha256']
    sample = decode_physics_prefix(raw, row['received_monotonic_seconds'])
    assert sample.packet_id == row['packet_id']
    assert json.loads(json.dumps(dict(sample.fields))) == row['fields']
    assert 0 <= sample.fields['speedKmh'] < 10
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--counterfactuals', action='store_true',
                        help='Offline full-mode yaw and scene-command interventions only')
    parser.add_argument('--fast-plan',
                        help='Explicit offline fast-path hypotheses, no live control')
    parser.add_argument('--fast-reference', help='Versioned motor-reference artifact')
    args = parser.parse_args()
    root = Path(args.recording)
    source, manifest = read(root / 'result.json'), read(root / 'manifest.json')
    plan = manifest['config']['plan']
    trial_index = manifest['config']['arguments']['perturbation_trial_index']
    trial = validate_perturbation_plan(plan, trial_index)
    if args.counterfactuals and trial['mode'] != 'full':
        raise ValueError('Counterfactual diagnosis requires original full-mode recording')
    if bool(args.fast_plan) != bool(args.fast_reference):
        raise ValueError('Paired explicit fast-path plan and motor reference required')
    if plan.get('diagnostic_transport_verification'):
        assert (verified_diagnostic_transport(plan)
                == manifest['config']['diagnostic_verification'])
    assert verified_yaw_trial(plan) == manifest['config']['yaw_verification']
    timed_trial = plan['schema'].endswith('.v2')
    if timed_trial:
        assert verified_timed_steering(plan) == manifest['config']['timed_verification']
    config, proof = verified_shadow(plan['inference_config'], plan['inference_verification'])
    fast_certificate = verified_fast_plan(plan)
    if fast_certificate != manifest['config'].get('fast_inference_certificate'):
        raise ValueError('Actual fast-stream certificate differs from recording')
    assert source['status'] == 'completed_physical_yaw_trial'
    assert source['released_neutral_and_pause_verified'] and not source['cleanup_errors']
    if timed_trial:
        assert not source['timed_worker']['error'] and not source['timed_worker_alive']
    assert manifest['config']['inference_certificate'] == proof
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, verification=proof,
        events_sha256=sha256(root / 'events.jsonl'),
        native_receipts_sha256=sha256(root / 'native-receipts.json'),
        bundle_sha256=sha256(root / 'vision/camera.zip'), task_speed_kmh=6,
        trial=trial, plan_sha256=manifest['config']['plan_sha256'],
        actuator_messages_sent=0, fast_inference_certificate=fast_certificate))
    result = dict(status='failed', goal_achieved=False, navigation_verified=False,
                  moving_stabilization_verified=False, actuator_messages_sent=0)
    try:
        if plan.get('preparation_engine_settle_seconds'):
            preparation = read(root/'engine-preparation.json')
            assert plan['preparation_engine_settle_seconds'] == 2.
            assert preparation['finished']-preparation['started'] >= 2.
            assert preparation['actor_observations'] == preparation['hidden_resets'] == 0
            assert preparation['initial_episode_state_preserved']
            assert len(preparation['native_samples']) >= 2
            times = []
            for record in preparation['native_samples']:
                sample = native(record)
                assert abs(sample.fields['speedKmh']) < .1
                assert sample.fields['gas'] < .02 and sample.fields['brake'] > .98
                times.append(sample.received_monotonic_seconds)
            assert np.all(np.diff(times) > 0)
            assert preparation['started'] <= times[0] < times[-1] <= preparation['finished']
            write_json(run/'engine-preparation-verification.json', dict(
                source_sha256=sha256(root/'engine-preparation.json'), samples=len(times),
                original_native_bytes_verified=True, observations_created=0))
        events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
        rows = [r for r in events if r['kind'] == 'flight_axis_observation']
        inputs = [r for r in events if r['kind'] == 'actor_input']
        assert len(rows) == len(inputs) == source['observations'] == 257
        if plan.get('preparation_engine_settle_seconds'):
            assert rows[0]['decision_monotonic_seconds'] > preparation['finished']
        driver_events = [e for e in events if 'request' in e
                         and not (timed_trial and e['kind'] == 'flight_axis_observation')]
        requests = [e['request'] for e in driver_events]
        assert len(requests) == source['actuator_messages_sent']
        for i, request in enumerate(requests):
            assert request['request_sequence'] == i+1 and request['driver_accepted']
            c = request['requested_control']
            assert abs(c['steering']) <= .1 and 0 <= c['throttle'] <= .12
            assert 0 <= c['brake'] <= 1 and not (c['throttle'] and c['brake'])
            assert request['driver_axis_values'] == {
                '48': round((c['steering']+1)/2*32767),
                '49': round(c['throttle']*32767), '50': round(c['brake']*32767)}
        by_sequence = {e['request']['request_sequence']: e for e in driver_events}
        if timed_trial:
            latest, active = None, 0.
            sources = {min(v['render_started_lower'] for v in row['camera']['views'])
                       for row in rows}
            for event in driver_events:
                kind = event['kind']
                if not kind.startswith('timed_'):
                    continue
                if kind == 'timed_base_request':
                    latest = event
                elif kind in ('timed_pulse_on', 'timed_pulse_off'):
                    assert event['base_control'] == latest['base_control']
                    assert event['source_render_started_lower'] == latest[
                        'source_render_started_lower']
                if kind == 'timed_pulse_on':
                    active = trial['sign']*.05
                elif kind in ('timed_pulse_off', 'timed_cancel_brake'):
                    active = 0.
                assert event['disturbance'] == active
                if not event['safe_override']:
                    lower = event['source_render_started_lower']
                    assert lower in sources
                    age = event['request']['driver_call_started_perf_counter_seconds']-lower
                    assert 0 <= age <= .15
        receipts = read(root / 'native-receipts.json')
        report = read(root / 'native-receipts-result.json')
        assert report['file_sha256'] == sha256(root / 'native-receipts.json')
        assert (report['status'] == 'completed' and not report['error']
                and not report['worker_alive'])
        assert len(receipts) == report['samples'] == source['native_receipts']['samples']
        for row in receipts:
            native(row)
        assert np.all(np.diff([r['packet_id'] for r in receipts]) > 0)
        assert np.all(np.diff([r['received_monotonic_seconds'] for r in receipts]) > 0)
        diagnostics = [e for e in events if e['kind'] == 'diagnostic']
        origin, session_id = None, None
        for d in diagnostics:
            sample = native(d['native'])
            raw = base64.b64decode(d['sdk']['wire_base64'], validate=True)
            assert hashlib.sha256(raw).hexdigest() == d['sdk']['wire_sha256']
            packet = decode_telemetry(raw, d['sdk']['received_monotonic_seconds'])
            if session_id is None:
                session_id = packet.raw['producer_session_id']
            assert packet.raw['producer_session_id'] == session_id
            assert abs(packet.received_monotonic_seconds-sample.received_monotonic_seconds) <= .05
            namespace = packet.raw['privileged_teacher']
            assert namespace['actor_input'] is False and d['actor_input'] is False
            truth = namespace['raw']
            xy = np.array([truth['world_position_x'], truth['world_position_y']])
            if origin is None:
                origin = xy
            np.testing.assert_array_equal(xy, d['position'])
            assert abs(np.linalg.norm(xy-origin)-d['displacement_m']) < 1e-12
            assert d['displacement_m'] < 20 and d['tyres_out'] == 0
        spool = read(root / 'event-spool.json')
        assert sha256(root / 'events.jsonl') == spool['sha256'] == sha256(spool['source'])
        torch.set_num_threads(1)
        torch.manual_seed(proof['initialization_seed'])
        stream = (FastFlightStream(plan['fast_stream_config'],
                                  native_identity=read(root/'identity.json'), episode_id=root.name)
                  if fast_certificate else FlightStream(plan['inference_config'],
                      native_identity=read(root/'identity.json'), episode_id=root.name))
        apply_yaw_mode(stream, trial['mode'])
        if not fast_certificate and plan.get('visual_backend') == 'frozen_visual_tail':
            install_frozen_visual_tail(stream)
        if not fast_certificate and plan.get('visual_backend') == 'graphed_visual_tail':
            install_graphed_visual_tail(stream)
        stream.goals[0, 1] = .2
        alternatives = FlightCounterfactualReplay(stream) if args.counterfactuals else None
        fast_replay = (FastOriginalReplay(stream, args.fast_plan, args.fast_reference)
                       if args.fast_plan else None)
        maximum, predictions, clocks = 0., [], []
        with torch.no_grad(), CameraArchive(root / 'vision') as camera:
            camera.verify_all()
            frame_count = 0
            for name in camera.by_name:
                if re.fullmatch(r'frame-\d{4}\.json', name):
                    validate_frame_aid_state(camera.json(name))
                    frame_count += 1
            assert frame_count == source['vision']['frames']
            cursor = 0
            for phase_index, phase in enumerate(plan['stages']):
                stream.goals[0, 0] = phase['task_turn_bias']
                for phase_sample in range(phase['samples']):
                    row, original = rows[cursor], inputs[cursor]
                    assert row['phase_index'] == phase_index and row['phase_sample'] == phase_sample
                    assert row['task_turn_bias'] == phase['task_turn_bias']
                    assert row['yaw_trial'] == trial
                    for key in ('camera', 'native', 'decision_monotonic_seconds', 'task_turn_bias'):
                        assert row[key] == original[key]
                    assert row['was_foreground'] is False
                    sample = native(row['native'])
                    native(row['native_before_request'])
                    candidate, pixels = row['camera'], {}
                    validate_frame_aid_state(candidate)
                    frame_raw = camera.read(Path(candidate['file']).name)
                    assert hashlib.sha256(frame_raw).hexdigest() == candidate['file_sha256']
                    frame = json.loads(frame_raw)
                    assert frame['index'] == candidate['capture_id']
                    verify_shared_frame(frame, candidate,
                                        camera.json(f'ack-{candidate["capture_id"]:04d}.json'))
                    for view in candidate['views']:
                        raw = camera.read(Path(view['file']).name)
                        assert hashlib.sha256(raw).hexdigest() == view['file_sha256']
                        rgb = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
                        assert hashlib.sha256(rgb.tobytes()).hexdigest() == view['rgb_sha256']
                        pixels[view['view_id']] = rgb
                    clock = row['decision_monotonic_seconds']
                    response = stream.advance(clock=clock, camera=candidate, pixels=pixels,
                        native_sample=decode_virtual_body_prefix(sample.prefix_bytes,
                            sample.received_monotonic_seconds))
                    actual = row['inference']
                    np.testing.assert_allclose(response['intent'], actual['intent'],
                                               atol=3e-6, rtol=2e-5)
                    maximum = max(maximum, float(np.max(np.abs(
                        np.asarray(response['intent'])-actual['intent']))))
                    for key in ('steps', 'visual_ticks', 'flight_ticks', 'decision_clock',
                                'capture_id', 'generation', 'recurrent_state_reset'):
                        assert actual[key] == response[key]
                    if fast_certificate:
                        for key in ('fast_stream_schema', 'fast_stream_config_sha256',
                                    'fast_hypothesis', 'fast_reference_sha256', 'fast_yaw_mode'):
                            assert actual[key] == response[key]
                    np.testing.assert_array_equal(stream.last_body.values, row['body_values'])
                    np.testing.assert_array_equal(stream.last_body.valid, row['body_valid'])
                    if alternatives is not None:
                        alternatives.observe(stream, clock, response['intent'])
                    if fast_replay is not None:
                        fast_replay.observe(stream, clock, response['intent'])
                    left = actual['intent'][0]
                    speed = sample.fields['speedKmh']
                    stop = phase['name'] in ('stationary_prefix', 'braking')
                    throttle = 0. if stop else float(np.clip(.03+.025*(6-speed), 0., .12))
                    brake = 1. if stop else float(np.clip(.03*(speed-6.4), 0., .2))
                    if brake > 0:
                        throttle = 0.
                    base_steering = 0. if stop else float(np.clip(-left*.6, -.1, .1))
                    driver = by_sequence[row['request']['request_sequence']]
                    assert driver['request'] == row['request']
                    pulse = (driver['disturbance'] if timed_trial else
                             trial['sign']*.05 if phase['name'] == 'perturbation' else 0.)
                    expected = dict(steering=float(np.clip(base_steering+pulse, -.1, .1)),
                                    throttle=throttle, brake=brake)
                    assert row['perturbation']['disturbance'] == pulse
                    assert row['perturbation']['base_control']['steering'] == base_steering
                    assert row['perturbation']['requested_control'] == expected
                    assert expected == row['request']['requested_control']
                    start = row['request']['driver_call_started_perf_counter_seconds']
                    assert 0 <= start-min(v['render_started_lower']
                                          for v in candidate['views']) <= .15
                    predictions.append(response['intent'])
                    clocks.append(clock)
                    cursor += 1
        assert cursor == 257
        full = np.array(predictions)
        phase_summaries, cursor = [], 0
        for phase in plan['stages']:
            stop = cursor+phase['samples']
            tail = slice(max(cursor, stop-10), stop)
            yaw = float(np.mean([r['body_values'][4] for r in rows[tail]]))
            phase_summaries.append(dict(phase=phase['name'], mean_yaw_left_rad_s=yaw,
                mean_speed_kmh=float(np.mean(
                    [r['native']['fields']['speedKmh'] for r in rows[tail]])),
                mean_steering_request=float(np.mean(
                    [r['request']['requested_control']['steering'] for r in rows[tail]])),
                mean_intent_left=float(full[tail, 0].mean())))
            cursor = stop
        final = native(read(root / 'native-after-release.json'))
        assert final.fields['speedKmh'] < .1 and final.fields['brake'] > .98
        assert abs(final.fields['steerAngle']) < .005
        assert read(root / 'api-pause/result.json')['status'] == 'completed'
        intervals = np.diff(clocks)
        np.savez_compressed(run / 'yaw-replay.npz', full=full, original_clock=np.array(clocks))
        if fast_replay is not None:
            np.savez_compressed(run/'fast-hypotheses.npz',
                intents=np.array(fast_replay.rows), original_clock=np.array(clocks),
                yaw=np.array([r['body_values'][4] for r in rows]),
                phase_index=np.array([r['phase_index'] for r in rows]))
            result['fast_original_replay'] = dict(contract=fast_replay.contract,
                hypotheses=fast_replay.hypotheses, scenarios=['full', 'zero_yaw'],
                maximum_original_neuron_error=fast_replay.maximum_state_error,
                maximum_original_intent_error=fast_replay.maximum_intent_error,
                original_observations=257, numerical_resets=0,
                archive_sha256=sha256(run/'fast-hypotheses.npz'))
        if alternatives is not None:
            np.savez_compressed(run / 'counterfactuals.npz', modes=np.array(MODES),
                intents=np.array(alternatives.rows), original_clock=np.array(clocks),
                yaw=np.array([r['body_values'][4] for r in rows]),
                phase_index=np.array([r['phase_index'] for r in rows]))
            result['counterfactual_diagnosis'] = dict(modes=MODES,
                full_batch_maximum_error=alternatives.maximum_error,
                original_observations=257, numerical_resets=0, actuator_messages_sent=0,
                physical_counterfactual_trajectory=False,
                scene_intervention='Existing normalized scene_command lesion, bias retained',
                archive_sha256=sha256(run/'counterfactuals.npz'))
        pulse_rows = [r for r in rows if r['phase'] == 'perturbation']
        recovery_rows = [r for r in rows if r['phase'] == 'recovery']
        if timed_trial:
            on = [e for e in events if e['kind'] == 'timed_pulse_on']
            off = [e for e in events if e['kind'] == 'timed_pulse_off']
            assert len(on) == len(off) == 1
            pulse_on = on[0]['request']['driver_call_finished_perf_counter_seconds']
            pulse_off = off[0]['request']['driver_call_finished_perf_counter_seconds']
            assert off[0]['scheduled_pulse_off'] == pulse_on+.28
        else:
            pulse_on = pulse_rows[0]['request']['driver_call_finished_perf_counter_seconds']
            pulse_off = recovery_rows[0]['request']['driver_call_finished_perf_counter_seconds']
        native_times = [r['received_monotonic_seconds'] for r in receipts]
        native_yaw = [r['fields']['localAngularVel'][1] for r in receipts]
        metrics = yaw_metrics(native_times, native_yaw, pulse_on=pulse_on, pulse_off=pulse_off,
                              spec=plan['evaluation'])
        baseline_speed = [r['fields']['speedKmh'] for r in receipts
                          if pulse_on-.3 <= r['received_monotonic_seconds'] < pulse_on]
        pulse_clipped = (any(e['clipped'] for e in driver_events
                            if e['kind'].startswith('timed_') and e['disturbance'])
                         if timed_trial else any(r['perturbation']['clipped'] for r in pulse_rows))
        result.update(status='completed_physical_yaw_trial_audit', observations=257,
            intent_max_error=maximum, original_full_history_replayed=True, hidden_resets=0,
            original_interval_seconds_max=float(intervals.max()),
            intervals_above_80ms=int((intervals > .08).sum()),
            camera_frames_aid_off_verified=frame_count, independent_native_receipts=len(receipts),
            maximum_speed_kmh=max(r['fields']['speedKmh'] for r in receipts),
            maximum_displacement_m=max(d['displacement_m'] for d in diagnostics),
            phase_summaries=phase_summaries,
            trial=trial, trial_index=trial_index, physical_metrics=metrics,
            pulse_on_host_clock=pulse_on, pulse_off_host_clock=pulse_off,
            timed_pulse=timed_trial, pulse_width_error_seconds=pulse_off-pulse_on-.28,
            pulse_timing_passed=abs(pulse_off-pulse_on-.28) <= .01,
            initial_position=origin.tolist(), baseline_speed_kmh=float(np.mean(baseline_speed)),
            pulse_clipped=pulse_clipped, actual_vehicle_condition_applied=True,
            learned_longitudinal_control_verified=False,
            released_centered_brake_hold_and_pause_verified=True,
            processing_seconds_percentiles=np.percentile(
                [r['inference']['processing_seconds'] for r in rows], [50, 95, 100]).tolist())
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
