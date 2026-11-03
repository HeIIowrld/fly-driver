"""Deploy tested bounded recovery while paused; bind the residual to the failed teacher."""
import json
import os
import shutil
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.win_process_receipt import ProcessReceipt
from scripts.record_ac_wgc import physics_record

base = Path(os.environ['LOCALAPPDATA'])/'flybrain'
failure = base/'runtime-runs/20261029-t1394-teacher-corpus-v3/episode-02-development/control'
read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
run = begin_run(base/'runtime-runs/20261029-t1401-brake-settle-deployment',
    dict(scope=__doc__, script_sha256=sha256(__file__), tests_passed=192,
         failed_control_result_sha256=sha256(failure/'result.json')))
shutil.copyfile(__file__, run/'executed-program.py')
result = dict(status='failed', installed=False, ui_actions_sent=0,
              lifecycle_requests_sent=0, actuator_messages_sent=0, goal_achieved=False)
handle = None
try:
    assert read(failure/'result.json')['reason'] == 'Independent native stationary brake hold not verified'
    old_native = read(failure/'native-after.json')
    api = WindowsACWindow()
    windows = api.discover()
    assert len(windows) == 1
    window = windows[0]
    assert window.pid == 28428 and api.user.GetForegroundWindow() != window.hwnd
    before, modified = read_published_json(base/'csp-session/status.json')
    assert 0 <= time.time()-modified < 2
    assert before['paused'] and not before['ai_controlled'] and not before['camera_active']
    assert not before['expert_probe_active'] and not before['is_online']
    assert not before.get('error') and not before.get('pending_receipt_id')
    assert abs(before['speed_kmh']) < .1 and before['gas'] < .02 and before['brake'] > .98
    with NativePhysicsReader(timeout=.15) as reader:
        native = physics_record(reader.receive())
    assert native['packet_id'] == old_native['packet_id']
    assert native['prefix_sha256'] == old_native['prefix_sha256']
    assert .1 <= native['fields']['speedKmh'] < .2
    assert native['fields']['gas'] < .02 and native['fields']['brake'] > .98
    axis = read('configs/ac-axis-probe.json')
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    assert sdk.GetVJDStatus(axis['device_id']) == 1
    target = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    old_source = base/'runtime-runs/20261029-t1375-origin-diagnostic-deployment/installed-after.lua'
    assert sha256(target) == sha256(old_source)
    source = Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    shutil.copyfile(target, run/'installed-before.lua')
    shutil.copyfile(source, run/'installed-after.lua')
    write_json(run/'before.json', dict(session=before, native=native,
        code_only_deployment_does_not_qualify_native_stationary_state=True))
    handle = ProcessReceipt(window.pid)
    temporary = target.with_name(target.name+'.flybrain-t1401.tmp')
    with temporary.open('xb') as stream:
        stream.write(source.read_bytes())
    assert sha256(temporary) == sha256(source)
    os.replace(temporary, target)
    result['installed'] = True
    started = time.monotonic()
    while time.monotonic()-started < 10:
        assert handle.snapshot()['active']
        after, modified = read_published_json(base/'csp-session/status.json')
        if (0 <= time.time()-modified < 2 and after['generation'] != before['generation']
                and after['capabilities'].get('brake_settle_after_teacher')):
            assert after['paused'] and not after['ai_controlled'] and not after['camera_active']
            assert api.user.GetForegroundWindow() != window.hwnd
            result.update(status='same_engine_bounded_recovery_deployed', session=after,
                          engine_pid=window.pid, installed_sha256=sha256(target))
            break
        time.sleep(.1)
    else:
        raise TimeoutError('Reload not observed; do not restart engine automatically')
except Exception as error:
    result.update(error=type(error).__name__, reason=str(error))
    raise
finally:
    if handle:
        handle.close()
    write_json(run/'result.json', result)
print(json.dumps(result), flush=True)
