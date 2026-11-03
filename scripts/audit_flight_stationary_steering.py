"""Full original neural replay and independent native steering response; stationary only."""

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
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.probe_flight_stationary_steering import validate_plan
from scripts.record_ac_engine_ai_probe import verified_shadow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def native(row):
    raw = base64.b64decode(row['prefix_base64'], validate=True)
    assert hashlib.sha256(raw).hexdigest() == row['prefix_sha256']
    sample = decode_physics_prefix(raw, row['received_monotonic_seconds'])
    assert sample.packet_id == row['packet_id']
    assert json.loads(json.dumps(dict(sample.fields))) == row['fields']
    f = sample.fields
    assert abs(f['speedKmh']) < .1 and f['gas'] < .02 and f['brake'] > .98
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    manifest = read(root / 'manifest.json')
    plan = manifest['config']['plan']
    validate_plan(plan)
    config, proof = verified_shadow(plan['inference_config'], plan['inference_verification'])
    source = read(root / 'result.json')
    assert source['status'] == 'completed_component_samples' and source['observations'] == 100
    assert source['released_neutral_and_pause_verified'] and not source['cleanup_errors']
    assert manifest['config']['inference_certificate'] == proof
    # Match replay and recording inference sources, not merely today's certificate.
    certificate_sources = read(
        Path(plan['inference_verification']) / 'manifest.json')['source_sha256']
    for file in certificate_sources:
        if file.startswith('src/') and file in manifest['source_sha256']:
            # Some non-inference sources legitimately changed after the certificate.
            if file.rsplit('/', 1)[-1][:-3] in (
                'flight_stream', 'flyvis_camera', 'flyvis_compat', 'virtual_body',
                'flight_circuit', 'flight_commands', 'flight_navigation', 'vehicle_circuit',
                'flight_learning', 'ac_csp_vision', 'ac_csp_geometry', 'ac_native_physics',
                'sensor_contract'):
                assert manifest['source_sha256'][file] == certificate_sources[file]
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, verification=proof,
        source_manifest_sha256=sha256(root / 'manifest.json'),
        events_sha256=sha256(root / 'events.jsonl'),
        native_receipts_sha256=sha256(root / 'native-receipts.json'),
        bundle_sha256=sha256(root / 'vision/camera.zip'),
        response_tolerance_native_game_units=.005,
        response_tolerance_role='post-collection engineering check, not held-out physiology',
        actuator_messages_sent=0))
    result = dict(status='failed', goal_achieved=False, navigation_verified=False,
                  physiological_equivalence_verified=False, actuator_messages_sent=0)
    try:
        events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
        rows = [r for r in events if r['kind'] == 'flight_axis_observation']
        inputs = [r for r in events if r['kind'] == 'actor_input']
        assert len(rows) == len(inputs) == 100
        assert sha256(root / 'events.jsonl') == read(root / 'event-spool.json')['sha256']
        assert sha256(read(root / 'event-spool.json')['source']) == sha256(root / 'events.jsonl')
        receipts = read(root / 'native-receipts.json')
        report = read(root / 'native-receipts-result.json')
        assert report['file_sha256'] == sha256(root / 'native-receipts.json')
        assert (report['status'] == 'completed' and not report['worker_alive']
                and not report['error'])
        assert len(receipts) == report['samples'] == source['native_receipts']['samples']
        for row in receipts:
            native(row)
        assert np.all(np.diff([r['packet_id'] for r in receipts]) > 0)
        receipt_times = np.array([r['received_monotonic_seconds'] for r in receipts])
        assert np.all(np.diff(receipt_times) > 0)
        assert report['started_at'] <= receipt_times[0] < receipt_times[-1] <= report['stopped_at']
        requests = [r['request'] for r in events if 'request' in r]
        assert len(requests) == source['actuator_messages_sent'] == 107
        for i, request in enumerate(requests):
            assert request['request_sequence'] == i+1 and request['driver_accepted']
            c = request['requested_control']
            assert c['throttle'] == 0 and c['brake'] == 1 and abs(c['steering']) <= .1
            # Independent affine encoder for this profile and measured vJoy range.
            assert request['driver_axis_values'] == {
                '48': round((c['steering']+1)/2*32767), '49': 0, '50': 32767}
            assert not request['driver_pressed_buttons_zero_based']
        assert np.all(np.diff([r['driver_call_started_perf_counter_seconds']
                               for r in requests]) > 0)
        torch.set_num_threads(1)
        torch.manual_seed(proof['initialization_seed'])
        stream = FlightStream(plan['inference_config'],
                              native_identity=read(root / 'identity.json'), episode_id=root.name)
        maximum, predictions, phases = 0., [], []
        with CameraArchive(root / 'vision') as camera:
            camera.verify_all()
            aid_frames = 0
            for name in camera.by_name:
                if re.fullmatch(r'frame-\d{4}\.json', name):
                    validate_frame_aid_state(camera.json(name))
                    aid_frames += 1
            assert aid_frames == source['vision']['frames']
            for index, (row, actor_input) in enumerate(zip(rows, inputs, strict=True)):
                phase = plan['stages'][index//25]
                assert row['phase_index'] == index//25 and row['phase_sample'] == index % 25
                assert row['task_turn_bias'] == phase['task_turn_bias']
                for key in ('camera', 'native', 'decision_monotonic_seconds', 'task_turn_bias'):
                    assert actor_input[key] == row[key]
                assert row['was_foreground'] is False
                sample = native(row['native'])
                native(row['native_before_request'])
                pixels, candidate = {}, row['camera']
                validate_frame_aid_state(candidate)
                frame_bytes = camera.read(Path(candidate['file']).name)
                assert hashlib.sha256(frame_bytes).hexdigest() == candidate['file_sha256']
                frame = json.loads(frame_bytes)
                assert frame['index'] == candidate['capture_id']
                assert frame['generation'] == candidate['generation']
                verify_shared_frame(frame, candidate,
                                    camera.json(f'ack-{candidate["capture_id"]:04d}.json'))
                for view in candidate['views']:
                    raw = camera.read(Path(view['file']).name)
                    assert hashlib.sha256(raw).hexdigest() == view['file_sha256']
                    rgb = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
                    assert hashlib.sha256(rgb.tobytes()).hexdigest() == view['rgb_sha256']
                    pixels[view['view_id']] = rgb
                stream.goals[0, 0] = phase['task_turn_bias']
                response = stream.advance(clock=row['decision_monotonic_seconds'], camera=candidate,
                    pixels=pixels, native_sample=decode_virtual_body_prefix(sample.prefix_bytes,
                        sample.received_monotonic_seconds))
                actual = row['inference']
                np.testing.assert_allclose(response['intent'], actual['intent'],
                                           atol=3e-6, rtol=2e-5)
                maximum = max(maximum, float(np.max(np.abs(
                    np.asarray(response['intent'])-actual['intent']))))
                for key in ('steps', 'visual_ticks', 'flight_ticks', 'decision_clock', 'capture_id',
                            'generation', 'recurrent_state_reset'):
                    assert actual[key] == response[key]
                left, drive = actual['intent']
                expected = dict(steering=float(np.clip(-left*.6, -.1, .1)), throttle=0., brake=1.)
                assert row['codec']['canonical_control'] == [-left, max(drive, 0), max(-drive, 0)]
                assert row['codec']['requested_control'] == row['request']['requested_control']
                assert expected == row['request']['requested_control']
                started = row['request']['driver_call_started_perf_counter_seconds']
                assert actual['output_host_clock'] <= started
                age = started-min(v['render_started_lower'] for v in candidate['views'])
                assert 0 <= age <= .15 and actual['actuator_messages_sent'] == 0
                predictions.append(response['intent'])
        # Complete original history above; tails below are response summaries only.
        calibration = Path(plan['calibration_reference'])
        old = [json.loads(s) for s in (calibration / 'samples.jsonl').read_text().splitlines()]
        old = [r for r in old if 'stage' in r]
        old_result = read(calibration / 'result.json')
        axis = manifest['config']['axis_configuration']
        assert len(old) == old_result['samples'] == 32 and old_result['status'] == 'completed'
        for key in ('profile_sha256', 'dll_sha256', 'device_id', 'bindings'):
            assert old_result[key] == axis[key]
        for row in old:
            raw = base64.b64decode(row['native']['prefix_base64'], validate=True)
            assert hashlib.sha256(raw).hexdigest() == row['native']['prefix_sha256']
            sample = decode_physics_prefix(raw, row['native']['received_monotonic_seconds'])
            assert sample.fields['steerAngle'] == row['native']['fields']['steerAngle']
        old_medians = [np.median([native_value['native']['fields']['steerAngle']
                                 for native_value in old if native_value['stage'] == stage
                                 and native_value['index'] >= 2]) for stage in range(4)]
        gain = float((old_medians[1]-old_medians[2])/(2*read(
            calibration / 'result.json')['request_amplitude']))
        for p, stage in enumerate(plan['stages']):
            tail = rows[p*25+15:(p+1)*25]
            lower = tail[0]['request']['driver_call_finished_perf_counter_seconds']+.06
            upper = tail[-1]['request']['driver_call_started_perf_counter_seconds']
            feedback = [r['fields']['steerAngle'] for r in receipts
                        if lower < r['received_monotonic_seconds'] < upper]
            assert len(feedback) >= 5
            request = float(np.median(
                [r['request']['requested_control']['steering'] for r in tail]))
            observed = float(np.median(feedback))
            error = abs(observed-request*gain)
            assert error < .005
            phases.append(dict(phase=stage['name'], request_median=request,
                native_median=observed, native_receipt_count=len(feedback),
                expected_native_from_prior_calibration=request*gain, absolute_native_error=error))
        assert phases[1]['native_median'] < -.02 and phases[2]['native_median'] > .02
        assert abs(phases[3]['native_median']-phases[0]['native_median']) < .005
        final = native(read(root / 'native-after-release.json'))
        assert abs(final.fields['steerAngle']) <= .005
        assert read(root / 'api-pause/result.json')['status'] == 'completed'
        intervals = np.diff([r['decision_monotonic_seconds'] for r in rows])
        np.savez_compressed(run / 'replayed-intentions.npz', intent=np.array(predictions))
        result.update(status='completed_stationary_response_audit', observations=100,
            intent_max_error=maximum, original_full_history_replayed=True, hidden_resets=0,
            original_interval_seconds_max=float(intervals.max()),
            intervals_above_80ms=int((intervals > .08).sum()),
            camera_frames_aid_off_verified=aid_frames, independent_native_receipts=len(receipts),
            native_receipt_interval_max_seconds=float(np.diff(receipt_times).max()),
            steering_units='native_game_units; radians/degrees not independently established',
            prior_native_gain_per_normalized_request=gain, phase_summaries=phases,
            steering_sign_and_stationary_response_verified=True,
            actual_model_requests=100, total_driver_requests=107,
            released_centered_brake_hold_and_pause_verified=True,
            learned_longitudinal_control_verified=False, moving_balance_verified=False,
            all_native_samples_stationary_brake_held=True,
            processing_seconds_percentiles=np.percentile(
                [r['inference']['processing_seconds'] for r in rows], [50, 95, 100]).tolist())
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(dict(status=result['status'], intent_max_error=maximum, phases=phases)))


if __name__ == '__main__':
    main()
