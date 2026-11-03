"""Replay the complete moving component episode, with open-loop yaw interventions only."""

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
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.ac_transport import decode_telemetry
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.probe_flight_moving_body import validate_plan
from scripts.record_ac_engine_ai_probe import verified_shadow


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
    args = parser.parse_args()
    root = Path(args.recording)
    source, manifest = read(root / 'result.json'), read(root / 'manifest.json')
    plan = manifest['config']['plan']
    validate_plan(plan)
    config, proof = verified_shadow(plan['inference_config'], plan['inference_verification'])
    assert source['status'] == 'completed_moving_component_samples'
    assert source['released_neutral_and_pause_verified'] and not source['cleanup_errors']
    assert manifest['config']['inference_certificate'] == proof
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, verification=proof,
        events_sha256=sha256(root / 'events.jsonl'),
        native_receipts_sha256=sha256(root / 'native-receipts.json'),
        bundle_sha256=sha256(root / 'vision/camera.zip'), task_speed_kmh=6,
        counterfactuals=['body yaw value zero, validity unchanged',
                         'body yaw value sign reversed, validity unchanged'],
        interventions_role='offline original-input replay, not new physical trajectories',
        expected_feedback='full minus zero-yaw left intent opposes measured left yaw',
        minimum_measured_yaw_rad_s=.005, minimum_delta_intent=1e-6,
        actuator_messages_sent=0))
    result = dict(status='failed', goal_achieved=False, navigation_verified=False,
                  moving_stabilization_verified=False, actuator_messages_sent=0)
    try:
        events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
        rows = [r for r in events if r['kind'] == 'flight_axis_observation']
        inputs = [r for r in events if r['kind'] == 'actor_input']
        assert len(rows) == len(inputs) == source['observations'] == 225
        requests = [e['request'] for e in events if 'request' in e]
        assert len(requests) == source['actuator_messages_sent']
        for i, request in enumerate(requests):
            assert request['request_sequence'] == i+1 and request['driver_accepted']
            c = request['requested_control']
            assert abs(c['steering']) <= .1 and 0 <= c['throttle'] <= .12
            assert 0 <= c['brake'] <= 1 and not (c['throttle'] and c['brake'])
            assert request['driver_axis_values'] == {
                '48': round((c['steering']+1)/2*32767),
                '49': round(c['throttle']*32767), '50': round(c['brake']*32767)}
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
        stream = FlightStream(plan['inference_config'],
                              native_identity=read(root / 'identity.json'), episode_id=root.name)
        stream.goals[0, 1] = .2
        alternate_state = stream.flight.initial_state(2)
        held, previous, maximum = None, None, 0.
        predictions, alternatives, clocks = [], [], []
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
                    np.testing.assert_array_equal(stream.last_body.values, row['body_values'])
                    np.testing.assert_array_equal(stream.last_body.valid, row['body_valid'])
                    left = actual['intent'][0]
                    speed = sample.fields['speedKmh']
                    stop = phase['name'] in ('stationary_prefix', 'braking')
                    throttle = 0. if stop else float(np.clip(.03+.025*(6-speed), 0., .12))
                    brake = 1. if stop else float(np.clip(.03*(speed-6.4), 0., .2))
                    if brake > 0:
                        throttle = 0.
                    expected = dict(steering=0. if stop else float(np.clip(-left*.6, -.1, .1)),
                                    throttle=throttle, brake=brake)
                    assert expected == row['request']['requested_control']
                    start = row['request']['driver_call_started_perf_counter_seconds']
                    assert 0 <= start-min(v['render_started_lower']
                                          for v in candidate['views']) <= .15
                    if previous is not None:
                        elapsed, dt = clock-previous, stream.flight.spec['dynamics']['dt']
                        count = int(elapsed//dt)
                        intervals = [dt]*count
                        if elapsed-count*dt > 1e-12:
                            intervals.append(elapsed-count*dt)
                        for step in intervals:
                            alternate_state = stream.flight._advance_elapsed_tensor(
                                held, alternate_state, step)
                    alternative = stream.flight._read_state_tensor(alternate_state)[0].cpu().numpy()
                    body = torch.tensor(row['body_values'], device='cuda').repeat(2, 1)
                    body[0, 4], body[1, 4] = 0., -body[1, 4]
                    valid = torch.tensor(row['body_valid'], device='cuda').repeat(2, 1)
                    held = stream.flight._drive_tensor(stream.visual_zero.expand(2, -1),
                        stream.visual_valid.expand(2, -1), body, valid,
                        stream.goals.expand(2, -1), stream.last_features.expand(2, -1))
                    predictions.append(response['intent'])
                    alternatives.append(alternative)
                    clocks.append(clock)
                    previous, cursor = clock, cursor+1
        assert cursor == 225
        full, alt = np.array(predictions), np.array(alternatives)
        phase_summaries, cursor = [], 0
        for phase in plan['stages']:
            stop = cursor+phase['samples']
            tail = slice(stop-10, stop)
            yaw = float(np.mean([r['body_values'][4] for r in rows[tail]]))
            delta = float((full[tail, 0]-alt[tail, 0, 0]).mean())
            phase_summaries.append(dict(phase=phase['name'], mean_yaw_left_rad_s=yaw,
                mean_full_minus_zero_yaw_left_intent=delta,
                mean_speed_kmh=float(np.mean(
                    [r['native']['fields']['speedKmh'] for r in rows[tail]])),
                mean_steering_request=float(np.mean(
                    [r['request']['requested_control']['steering'] for r in rows[tail]])),
                yaw_excitation_sufficient=abs(yaw) >= .005,
                feedback_opposes_yaw=bool(abs(yaw) >= .005 and -np.sign(yaw)*delta > 1e-6)))
            cursor = stop
        final = native(read(root / 'native-after-release.json'))
        assert final.fields['speedKmh'] < .1 and final.fields['brake'] > .98
        assert abs(final.fields['steerAngle']) < .005
        assert read(root / 'api-pause/result.json')['status'] == 'completed'
        intervals = np.diff(clocks)
        np.savez_compressed(run / 'yaw-replay.npz', full=full, zero_yaw=alt[:, 0],
                            reversed_yaw=alt[:, 1], original_clock=np.array(clocks))
        result.update(status='completed_moving_body_replay_audit', observations=225,
            intent_max_error=maximum, original_full_history_replayed=True, hidden_resets=0,
            original_interval_seconds_max=float(intervals.max()),
            intervals_above_80ms=int((intervals > .08).sum()),
            camera_frames_aid_off_verified=frame_count, independent_native_receipts=len(receipts),
            maximum_speed_kmh=max(r['fields']['speedKmh'] for r in receipts),
            maximum_displacement_m=max(d['displacement_m'] for d in diagnostics),
            phase_summaries=phase_summaries,
            yaw_feedback_opposes_both_turns=all(phase_summaries[i]['feedback_opposes_yaw']
                                               for i in (2, 3)),
            offline_yaw_intervention_only=True, learned_longitudinal_control_verified=False,
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
