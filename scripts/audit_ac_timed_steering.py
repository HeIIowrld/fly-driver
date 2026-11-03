"""Independent original-byte audit of six finite camera-sourced stationary timed pulses."""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_vision import validate_frame_aid_state
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


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


def verified_timed_steering(plan):
    audit = Path(plan['timed_actuator_verification'])
    result, config = read(audit/'result.json'), read(audit/'manifest.json')['config']
    assert result['status'] == 'completed_timed_stationary_audit'
    assert result['all_six_width_and_native_responses_verified']
    assert result['maximum_width_error_seconds'] <= .01
    source = Path(config['arguments']['recording'])
    assert sha256(source/'manifest.json') == config['source_manifest_sha256']
    manifest = read(source/'manifest.json')
    assert manifest['config']['axis'] == read(plan['axis_config'])
    for module in ('ac_timed_pulse', 'win_deadline_wait', 'ac_vjoy'):
        file = 'src/flybrain_drive/'+module+'.py'
        assert manifest['source_sha256'][file] == sha256(file)
    return dict(audit=str(audit), result_sha256=sha256(audit/'result.json'),
                actuator_sha256=sha256('src/flybrain_drive/ac_timed_pulse.py'),
                waiter_sha256=sha256('src/flybrain_drive/win_deadline_wait.py'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.recording)
    source, manifest = read(root/'result.json'), read(root/'manifest.json')
    assert source['status'] == 'completed_timed_stationary_samples' and len(source['pulses']) == 6
    assert source['released_neutral_and_pause_verified'] and not source['worker_alive']
    assert not source['cleanup_errors']
    for file in ('ac_timed_pulse', 'win_deadline_wait', 'ac_vjoy'):
        path = 'src/flybrain_drive/'+file+'.py'
        assert manifest['source_sha256'][path] == sha256(path)
    calibration = Path('runs/ac-shared-lateral-recovery-prepare-20261029-t933/calibration')
    run = begin_run(a.output, dict(arguments=vars(a), scope=__doc__,
        events_sha256=sha256(root/'events.jsonl'),
        native_receipts_sha256=sha256(root/'native-receipts.json'),
        bundle_sha256=sha256(root/'vision/camera.zip'),
        calibration_samples_sha256=sha256(calibration/'samples.jsonl'),
        source_manifest_sha256=sha256(root/'manifest.json'), actuator_messages_sent=0))
    result = dict(status='failed', goal_achieved=False, actor_qualified=False,
                  navigation_verified=False, actuator_messages_sent=0)
    try:
        events = [json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
        spool = read(root/'event-spool.json')
        assert sha256(root/'events.jsonl') == spool['sha256'] == sha256(spool['source'])
        receipts = read(root/'native-receipts.json')
        report = read(root/'native-receipts-result.json')
        assert (report['status'] == 'completed' and not report['error']
                and not report['worker_alive'])
        assert report['file_sha256'] == sha256(root/'native-receipts.json')
        assert len(receipts) == report['samples'] == source['native_receipts']['samples']
        for row in receipts:
            native(row)
        assert np.all(np.diff([r['packet_id'] for r in receipts]) > 0)
        assert np.all(np.diff([r['received_monotonic_seconds'] for r in receipts]) > 0)
        latest, available, active, previous_calls = {}, {}, {}, {}
        requests = []
        with CameraArchive(root/'vision') as camera:
            camera.verify_all()
            frames = 0
            for name in camera.by_name:
                if re.fullmatch(r'frame-\d{4}\.json', name):
                    validate_frame_aid_state(camera.json(name))
                    frames += 1
            assert frames == source['vision']['frames']
            for event in events:
                kind = event['kind']
                if kind == 'camera_source':
                    row = event['camera']
                    validate_frame_aid_state(row)
                    native(event['native'])
                    raw = camera.read(Path(row['file']).name)
                    assert hashlib.sha256(raw).hexdigest() == row['file_sha256']
                    verify_shared_frame(json.loads(raw), row,
                                        camera.json(f'ack-{row["capture_id"]:04d}.json'))
                    available.setdefault(event['pulse_index'], set()).add(
                        min(v['render_started_lower'] for v in row['views']))
                if 'request' not in event:
                    continue
                request = event['request']
                requests.append(request)
                control = request['requested_control']
                assert request['request_sequence'] == len(requests) and request['driver_accepted']
                assert control['throttle'] == 0 and control['brake'] == 1
                assert abs(control['steering']) <= .05
                assert not request['driver_pressed_buttons_zero_based']
                assert request['driver_axis_values'] == {
                    '48': round((control['steering']+1)/2*32767), '49': 0, '50': 32767}
                if not kind.startswith('timed_'):
                    continue
                index = event['pulse_index']
                if not event['safe_override']:
                    src = event['source_render_started_lower']
                    assert src in available[index]
                    assert 0 <= request['driver_call_started_perf_counter_seconds']-src <= .15
                    assert not event['clipped']
                    assert event['accepted_command_lease_deadline'] == request[
                        'driver_call_finished_perf_counter_seconds']+.08
                    if index in previous_calls:
                        assert request['driver_call_started_perf_counter_seconds'] <= (
                            previous_calls[index]+.08)
                    previous_calls[index] = request['driver_call_finished_perf_counter_seconds']
                if kind == 'timed_base_request':
                    latest[index] = event
                elif kind in ('timed_pulse_on', 'timed_pulse_off'):
                    assert event['base_control'] == latest[index]['base_control']
                    assert event['source_render_started_lower'] == latest[index][
                        'source_render_started_lower']
                if kind == 'timed_pulse_on':
                    active[index] = .05 if index % 2 == 0 else -.05
                elif kind in ('timed_pulse_off', 'timed_cancel_brake'):
                    active[index] = 0.
                assert event['disturbance'] == active.get(index, 0.)
                assert control['steering'] == event['disturbance']
        assert len(requests) == source['actuator_messages_sent']
        old = [json.loads(s) for s in (calibration/'samples.jsonl').read_text().splitlines()]
        old = [r for r in old if 'stage' in r]
        for row in old:
            raw = base64.b64decode(row['native']['prefix_base64'], validate=True)
            assert hashlib.sha256(raw).hexdigest() == row['native']['prefix_sha256']
            sample = decode_physics_prefix(raw, row['native']['received_monotonic_seconds'])
            assert sample.fields['steerAngle'] == row['native']['fields']['steerAngle']
        medians = [np.median([r['native']['fields']['steerAngle'] for r in old
                             if r['stage'] == i and r['index'] >= 2]) for i in range(4)]
        gain = float((medians[1]-medians[2])/
                     (2*read(calibration/'result.json')['request_amplitude']))
        pulses = []
        for index in range(6):
            group = [e for e in events if e.get('pulse_index') == index]
            on = [e for e in group if e['kind'] == 'timed_pulse_on']
            off = [e for e in group if e['kind'] == 'timed_pulse_off']
            assert len(on) == len(off) == 1
            t0 = on[0]['request']['driver_call_finished_perf_counter_seconds']
            t1 = off[0]['request']['driver_call_finished_perf_counter_seconds']
            width = t1-t0
            assert abs(width-.28) <= .01
            assert width == source['pulses'][index]['actual_driver_width_seconds']
            assert source['pulses'][index]['worker']['wait_backend'] == (
                'win32_high_resolution_waitable_timer')
            schedule = next(e for e in group if e['kind'] == 'timed_pulse_schedule')
            assert schedule['deadline'] == t0+.28 == off[0]['scheduled_pulse_off']
            plateau = [r['fields']['steerAngle'] for r in receipts
                       if t0+.08 < r['received_monotonic_seconds'] < t1-.02]
            centered = [r['fields']['steerAngle'] for r in receipts
                        if t1+.08 < r['received_monotonic_seconds'] < t1+.30]
            assert len(plateau) >= 5 and len(centered) >= 5
            expected = (.05 if index % 2 == 0 else -.05)*gain
            observed, neutral = float(np.median(plateau)), float(np.median(centered))
            assert abs(observed-expected) < .005 and abs(neutral) < .005
            pulses.append(dict(index=index, driver_width_seconds=width,
                driver_width_error_seconds=width-.28, expected_native_game_units=expected,
                actual_native_game_units=observed, centered_native_game_units=neutral,
                plateau_samples=len(plateau), centered_samples=len(centered)))
        final = native(read(root/'native-after-release.json'))
        assert abs(final.fields['steerAngle']) < .005
        assert read(root/'api-pause/result.json')['status'] == 'completed'
        result.update(status='completed_timed_stationary_audit', pulses=pulses,
            all_six_width_and_native_responses_verified=True,
            maximum_width_error_seconds=max(abs(p['driver_width_error_seconds']) for p in pulses),
            native_receipts=len(receipts), camera_frames_aid_off_verified=frames,
            original_source_ages_not_redated=True, per_request_source_guard_seconds=.15,
            accepted_command_gap_limit_seconds=.08,
            driver_requests=len(requests), released_neutral_and_pause_verified=True,
            native_units='game signal scale; not independently established radians/degrees',
            neural_inference=False, moving_stabilization_verified=False,
            physical_application_time_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
