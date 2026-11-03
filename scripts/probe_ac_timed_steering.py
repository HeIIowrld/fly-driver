"""Stationary actual-camera timed steering; full brake throughout, no neural driving claim."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_native_receipts import NativeReceiptCollector
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_timed_pulse import TimedSteeringPulse
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def stationary(sample):
    f = sample.fields
    if abs(f['speedKmh']) >= .1 or f['gas'] >= .02 or f['brake'] <= .98:
        raise ValueError('Actual stationary full-brake hold required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    axis = read('configs/ac-axis-probe.json')
    periods = [.03, .05, .04, .06, .04, .03]
    run = begin_run(a.output, dict(arguments=vars(a), scope=__doc__, pulse_seconds=.28,
        pulse_amplitude=.05, update_periods=periods, signs=[1, -1, 1, -1, 1, -1],
        maximum_width_error_seconds=.01, stationary_native_tolerance=.005,
        axis=axis, actor_inference=False, neural_resets=0, ui_actions_sent=0))
    result = dict(status='failed', goal_achieved=False, actor_qualified=False,
                  neural_inference=False, ui_actions_sent=0, pulses=[])
    spool = Path(os.environ['LOCALAPPDATA'])/'flybrain/event-spool'/uuid.uuid4().hex
    spool.mkdir(parents=True, exist_ok=False)
    log = (spool/'events.jsonl').open('x', encoding='utf8')
    log_lock = threading.Lock()
    session = vision = reader = ai = collector = pulse = None
    resumed, cleanup = False, []
    root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'

    def emit(event):
        with log_lock:
            log.write(json.dumps(event, allow_nan=False)+'\n')
            log.flush()

    def lifecycle(command):
        output = run/('api-'+command)
        with (run/('api-'+command+'.log')).open('x', encoding='utf8') as pipe:
            child = subprocess.Popen([sys.executable, '-m', 'scripts.control_ac_session',
                '--command', command, '--execute', '--output', str(output)],
                stdout=pipe, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            child.wait()
        assert child.returncode == 0 and read(output/'result.json')['status'] == 'completed'

    try:
        native = WindowsACWindow()
        windows = native.discover()
        assert len(windows) == 1
        window = windows[0]
        state, modified = read_published_json(root/'status.json')
        assert (0 <= time.time()-modified <= 2 and state['paused'] and not state['ai_controlled']
                and not state['camera_active'] and state['ideal_line_shown'] is False)
        assert state['car_id'] == 'bmw_z4_gt3' and state['track_id'] == 'monza'
        assert not state['is_online'] and native.user.GetForegroundWindow() != window.hwnd
        assert sha256(axis['profile_path']) == axis['profile_sha256']
        controllers = inspect_vjoy_controllers()
        assert controllers['candidate_ac_joy_index'] == 0 and len(controllers['vjoy_devices']) == 1
        assert controllers['vjoy_devices'][0]['instance_guid'] == axis['instance_guid']
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        assert sdk.GetVJDStatus(axis['device_id']) == 1
        reader = NativePhysicsReader(timeout=.15, clock=time.perf_counter)
        ai = NativeTeacherStatusReader()
        stationary(reader.receive())
        assert not ai.snapshot()['builtin_ai_controlled']
        if not a.execute:
            result['status'] = 'read_only'
        else:
            lifecycle('resume')
            resumed = True
            stationary(reader.receive())
            collector = NativeReceiptCollector()
            collector.start()
            session = VJoySession(sdk, axis['device_id'],
                                  [AxisBinding(**b) for b in axis['bindings']])
            emit(dict(kind='driver_open', request=session.open()))
            vision = CSPVisionRecorder(window.hwnd, read('configs/ac-csp-yaw-retina.json'),
                max_frames=400, period=.04, max_bytes=512*1024**2,
                image_format='bmp', shared_transport=True)
            vision.wait_ready()
            for index, period in enumerate(periods):
                pulse_events = []

                def record(event, index=index, pulse_events=pulse_events):
                    pulse_events.append(event)
                    emit(dict(event, pulse_index=index))

                pulse = TimedSteeringPulse(session, record)
                start, on = time.perf_counter(), None
                while on is None or time.perf_counter() < (
                        on['request']['driver_call_finished_perf_counter_seconds']+.65):
                    began = time.perf_counter()
                    current, modified = read_published_json(root/'status.json')
                    assert (0 <= time.time()-modified <= 2
                            and current['generation'] == state['generation']
                            and current['epoch'] == state['epoch'] and not current['paused']
                            and not current['ai_controlled']
                            and current['ideal_line_shown'] is False)
                    assert not ai.snapshot()['builtin_ai_controlled']
                    assert native.probe(window.hwnd, require_foreground=False) == window
                    assert native.user.GetForegroundWindow() != window.hwnd
                    assert sha256(axis['profile_path']) == axis['profile_sha256']
                    collector.check()
                    vision.wait_for_candidate(timeout=.04)
                    sample = reader.receive()
                    stationary(sample)
                    camera = vision.candidate_before(sample.received_monotonic_seconds)
                    emit(dict(kind='camera_source', pulse_index=index, camera=camera,
                              native=physics_record(sample)))
                    pulse.submit(dict(steering=0., throttle=0., brake=1.),
                        source_render_started_lower=min(v['render_started_lower']
                                                       for v in camera['views']))
                    if on is None and time.perf_counter()-start >= .2:
                        on = pulse.begin(1 if index % 2 == 0 else -1)
                    time.sleep(max(0., period-(time.perf_counter()-began)))
                report = pulse.stop()
                assert not report['error']
                off = next(e for e in pulse_events if e['kind'] == 'timed_pulse_off')
                width = (off['request']['driver_call_finished_perf_counter_seconds']
                         - on['request']['driver_call_finished_perf_counter_seconds'])
                result['pulses'].append(dict(index=index, update_period=period,
                    actual_driver_width_seconds=width,
                    width_passed=abs(width-.28) <= .01, worker=report))
                pulse = None
            assert all(r['width_passed'] for r in result['pulses'])
            result['status'] = 'completed_timed_stationary_samples'
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if pulse is not None:
            try:
                result['last_worker'] = pulse.stop()
            except Exception as error:
                cleanup.append('timed worker: '+str(error))
        live_worker = pulse is not None and pulse.thread.is_alive()
        if session is not None and session.owned and not live_worker:
            try:
                emit(dict(kind='driver_close', request=session.close()))
                time.sleep(.2)
            except Exception as error:
                cleanup.append('driver: '+str(error))
        if collector is not None:
            try:
                result['native_receipts'] = collector.close()
                write_json(run/'native-receipts.json',
                           [physics_record(s) for s in collector.samples])
                write_json(run/'native-receipts-result.json', dict(result['native_receipts'],
                    file_sha256=sha256(run/'native-receipts.json')))
            except Exception as error:
                cleanup.append('native: '+str(error))
        if vision is not None:
            try:
                result['vision'] = vision.close()
            except Exception as error:
                cleanup.append('camera: '+str(error))
        if resumed and not live_worker:
            try:
                sample = reader.receive()
                stationary(sample)
                assert abs(sample.fields['steerAngle']) < .005
                assert sdk.GetVJDStatus(axis['device_id']) == 1
                write_json(run/'native-after-release.json', physics_record(sample))
                lifecycle('pause')
                result['released_neutral_and_pause_verified'] = True
            except Exception as error:
                cleanup.append('final state: '+str(error))
        if ai is not None:
            ai.close()
        if reader is not None:
            reader.close()
        log.close()
        shutil.copyfile(spool/'events.jsonl', run/'events.jsonl')
        assert sha256(spool/'events.jsonl') == sha256(run/'events.jsonl')
        write_json(run/'event-spool.json', dict(source=str(spool/'events.jsonl'),
                                               source_retained=True,
                                               sha256=sha256(run/'events.jsonl')))
        if vision is not None and not vision.thread.is_alive():
            result['vision_archive'] = vision.archive_to(run/'vision', bundle=True)
        result.update(cleanup_errors=cleanup, worker_alive=live_worker,
                      actuator_messages_sent=session.submit_attempts if session else 0)
        if cleanup or live_worker:
            result['status'] = 'failed'
        write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result.get('reason', str(cleanup)))


if __name__ == '__main__':
    main()
