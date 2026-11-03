"""Reset only the proven owned paused failed engine; cold start once without activation."""
import configparser
import ctypes
import json
import os
import shutil
import subprocess
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

read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
base = Path(os.environ['LOCALAPPDATA'])/'flybrain'
run = begin_run(base/'runtime-runs/20261029-t1403-owned-engine-reset',
    dict(scope=__doc__, script_sha256=sha256(__file__), tests_passed=193,
         reason='Finite brake recovery aborted; original failed episode remains immutable'))
shutil.copyfile(__file__, run/'executed-program.py')
result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
              launch_attempts=0, owned_engine_termination_attempts=0, goal_achieved=False)
old = None
try:
    assert read(base/'runtime-runs/20261029-t1318-background-cold-start/result.json')['pid'] == 28428
    assert read(base/'runtime-runs/20261029-t1402-brake-settle-recovery/result.json')['status'] == 'failed'
    api = WindowsACWindow()
    windows = api.discover()
    assert len(windows) == 1 and windows[0].pid == 28428
    window = windows[0]
    assert api.user.GetForegroundWindow() != window.hwnd
    state, modified = read_published_json(base/'csp-session/status.json')
    assert 0 <= time.time()-modified < 2 and state['paused']
    assert not state['ai_controlled'] and not state['camera_active'] and not state['expert_probe_active']
    assert not state.get('pending_receipt_id') and not state['is_online']
    with NativePhysicsReader(timeout=.15) as reader:
        native = physics_record(reader.receive())
    assert native['packet_id'] == 222778 and abs(native['fields']['speedKmh']) < 1
    assert native['fields']['gas'] < .02 and native['fields']['brake'] > .98
    axis = read('configs/ac-axis-probe.json')
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    assert sdk.GetVJDStatus(axis['device_id']) == 1
    game = Path(window.executable).parent
    assert game == Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa')
    cfg = Path(axis['profile_path']).parent
    video = configparser.ConfigParser(strict=False)
    video.read(cfg/'video.ini', encoding='utf8')
    assert video.getint('VIDEO', 'FULLSCREEN') == 0
    assert video.getint('VIDEO', 'WIDTH') == 640 and video.getint('VIDEO', 'HEIGHT') == 360
    target = game/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    assert sha256(target) == read(base/'runtime-runs/20261029-t1401-brake-settle-deployment/result.json')['installed_sha256']
    source = Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    shutil.copyfile(target, run/'installed-before.lua')
    shutil.copyfile(source, run/'installed-after.lua')
    write_json(run/'before.json', dict(session=state, native=native, engine_pid=window.pid,
        video_sha256=sha256(cfg/'video.ini'), race_sha256=sha256(cfg/'race.ini'),
        controls_sha256=sha256(axis['profile_path'])))
    old = ProcessReceipt(window.pid)
    assert old.snapshot()['active']
    # This is a simulation reset of the recorded owned process, not a resume or guard waiver.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    termination = kernel.OpenProcess(1, False, window.pid)
    assert termination
    result['owned_engine_termination_attempts'] = 1
    try:
        assert kernel.TerminateProcess(termination, 0)
    finally:
        kernel.CloseHandle(termination)
    deadline = time.monotonic()+10
    while old.snapshot()['active'] and time.monotonic() < deadline:
        time.sleep(.05)
    write_json(run/'original-process-terminal.json', old.snapshot())
    assert not old.snapshot()['active'] and not api.discover()
    temporary = target.with_name(target.name+'.flybrain-t1403.tmp')
    with temporary.open('xb') as stream:
        stream.write(source.read_bytes())
    assert sha256(temporary) == sha256(source)
    os.replace(temporary, target)
    environment = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    environment.update(SteamAppId='244210', SteamGameId='244210')
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 4
    result['launch_attempts'] = 1
    child = subprocess.Popen([str(game/'acs.exe')], cwd=game, env=environment,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        startupinfo=startup, creationflags=subprocess.DETACHED_PROCESS |
        subprocess.CREATE_NEW_PROCESS_GROUP | 0x01000000)
    result['new_engine_pid'] = child.pid
    write_json(run/'launch.json', dict(pid=child.pid, window='640x360', fullscreen=False,
                                      startup_show_window='SW_SHOWNOACTIVATE', attempts=1))
    first_healthy = None
    while True:
        if child.poll() is not None:
            raise RuntimeError('New owned engine exited; no automatic replacement')
        current = [w for w in api.discover() if w.pid == child.pid]
        if len(current) == 1:
            try:
                after, modified = read_published_json(base/'csp-session/status.json')
                healthy = (0 <= time.time()-modified < 2
                    and int(after['window_handle'].removesuffix('ULL')) == current[0].hwnd
                    and after['generation'] != state['generation'] and not after.get('error'))
                if healthy:
                    if first_healthy is None:
                        first_healthy = time.monotonic()
                    if time.monotonic()-first_healthy >= 3:
                        result.update(status='owned_reset_background_publisher_verified', session=after,
                            foreground_after=api.user.GetForegroundWindow() == current[0].hwnd)
                        break
            except (OSError, ValueError, KeyError):
                pass
        time.sleep(.2)  # Same owned process; never replace on an observation timeout.
except Exception as error:
    result.update(error=type(error).__name__, reason=str(error))
    raise
finally:
    if old:
        old.close()
    write_json(run/'result.json', result)
print(json.dumps(result), flush=True)
