"""One bounded corrective lease for the exact terminal teacher residual; never retry."""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader, repeated_snapshot
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record

read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
base = Path(os.environ['LOCALAPPDATA'])/'flybrain'
root = base/'csp-session'
failed = base/'runtime-runs/20261029-t1394-teacher-corpus-v3/episode-02-development/control'
run = begin_run(base/'runtime-runs/20261029-t1402-brake-settle-recovery',
    dict(scope=__doc__, script_sha256=sha256(__file__), normal_native_guard_unchanged=True,
         failed_control_sha256=sha256(failed/'result.json')))
shutil.copyfile(__file__, run/'executed-program.py')
result = dict(status='failed', lifecycle_requests_sent=0, actuator_messages_sent=0,
              ui_actions_sent=0, goal_achieved=False)
samples = []
reader = ai = None
try:
    old_native = read(failed/'native-after.json')
    old_receipt = read(failed/'api-receipt.json')
    assert read(failed/'result.json')['reason'] == 'Independent native stationary brake hold not verified'
    assert old_receipt == read(root/(old_receipt['id']+'.terminal.json'))
    assert old_receipt['status'] == 'api_returned_true'
    api = WindowsACWindow()
    windows = api.discover()
    assert len(windows) == 1 and windows[0].pid == 28428
    window = windows[0]
    assert api.user.GetForegroundWindow() != window.hwnd
    target = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    assert sha256(target) == sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    axis = read('configs/ac-axis-probe.json')
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    assert sdk.GetVJDStatus(axis['device_id']) == 1
    reader, ai = NativePhysicsReader(timeout=.15), NativeTeacherStatusReader()
    before_native = physics_record(reader.receive())
    assert before_native['prefix_sha256'] == old_native['prefix_sha256']
    assert before_native['packet_id'] == old_native['packet_id']
    native = before_native['fields']
    assert .1 <= native['speedKmh'] < .2 and native['gas'] < .02 and native['brake'] > .98
    assert abs(native['steerAngle']) < .005
    before, modified = read_published_json(root/'status.json')
    assert 0 <= time.time()-modified < 2 and not before.get('error')
    assert before['paused'] and not before['camera_active'] and not before['ai_controlled']
    assert not before['expert_probe_active'] and not before.get('pending_receipt_id')
    assert before['capabilities']['brake_settle_after_teacher']
    assert before['generation'] == read(base/'runtime-runs/20261029-t1401-brake-settle-deployment/result.json')['session']['generation']
    write_json(run/'before.json', dict(session=before, native=before_native, native_teacher=ai.snapshot()))
    request = dict(schema='flybrain.csp.session.request.v1', id=uuid.uuid4().hex,
        command='brake-settle-after-teacher', generation=before['generation'], epoch=before['epoch'],
        expires_clock=before['script_clock']+2.5, terminal_teacher_request_id=old_receipt['id'],
        native_residual_after_teacher_verified=True, native_residual_speed_kmh=native['speedKmh'])
    write_json(run/'request.json', request)
    write_json(root/'intent-journal'/(request['id']+'.json'),
               dict(request=request, failed_control_sha256=sha256(failed/'result.json'),
                    native_prefix_sha256=before_native['prefix_sha256'], publication_attempts=1))
    write_json(root/'request.json', request)
    result['lifecycle_requests_sent'] = 1
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        samples.append(physics_record(repeated_snapshot(reader.mapping, clock=time.perf_counter)))
        terminal = root/(request['id']+'.terminal.json')
        accepted = root/(request['id']+'.json')
        if terminal.exists():
            receipt = read(terminal)
            write_json(run/'api-receipt.json', receipt)
            assert receipt['id'] == request['id'] and receipt['command'] == request['command']
            assert receipt['status'] == 'api_returned_true' and receipt['brake_settle_passed']
            assert receipt['duration_seconds'] <= .36
            assert receipt['maximum_position_delta_m'] <= .02
            assert receipt['control_setters_sent'] == 0
            break
        if accepted.exists():
            assert read(accepted)['status'] == 'accepted'
        time.sleep(.005)
    else:
        raise TimeoutError('Original finite recovery outcome unknown; no retry')
    after_native = physics_record(repeated_snapshot(reader.mapping, clock=time.perf_counter))
    write_json(run/'native-after.json', after_native)
    assert after_native['packet_id'] > before_native['packet_id']
    after = after_native['fields']
    assert abs(after['speedKmh']) < .1 and after['gas'] < .02 and after['brake'] > .98
    assert all(s['fields']['gas'] < .02 and s['fields']['brake'] > .98
               and abs(s['fields']['speedKmh']) < .2 for s in samples)
    assert api.user.GetForegroundWindow() != window.hwnd
    result.update(status='completed_exact_teacher_residual_recovery',
        native_packet_before=before_native['packet_id'], native_packet_after=after_native['packet_id'],
        native_speed_after_kmh=after['speedKmh'], native_teacher_after=ai.snapshot(),
        duration_seconds=receipt['duration_seconds'], maximum_position_delta_m=receipt['maximum_position_delta_m'],
        normal_native_stationary_guard_passed=True, student_policy_applied=False)
except Exception as error:
    result.update(error=type(error).__name__, reason=str(error))
    raise
finally:
    if reader:
        reader.close()
    if ai:
        ai.close()
    write_json(run/'native-samples.json', samples)
    write_json(run/'result.json', result)
print(json.dumps(result), flush=True)
