"""Hash-pinned Lua replacement in a verified paused offline engine; no desktop input."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.control_ac_session import validate_heartbeat
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-installed-sha256', required=True)
    parser.add_argument('--expected-engine-pid', required=True, type=int)
    parser.add_argument('--expected-generation-before', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    refuse_leased_transport('deploy_ac_camera_service')
    source = Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862953,
                                     new_sha256=sha256(source)))
    result = dict(status='failed', installed=False, ui_actions_sent=0,
                  actuator_messages_sent=0, ai_api_calls=0, goal_achieved=False)
    try:
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1 or windows[0].pid != args.expected_engine_pid:
            raise ValueError('Exactly one original engine required')
        window = windows[0]
        status = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'

        def checked_status(*, require_original_generation=False):
            current = native.discover()
            if (len(current) != 1 or current[0].pid != args.expected_engine_pid
                    or current[0].hwnd != window.hwnd
                    or native.user.GetForegroundWindow() == window.hwnd):
                raise ValueError('Exact original background engine required before replacement')
            refuse_leased_transport('deploy_ac_camera_service', status_path=status)
            state = json.loads(status.read_text(encoding='utf8'))
            validate_heartbeat(state, time.time()-status.stat().st_mtime, mutating=True)
            if (state.get('epoch') != 0 or require_original_generation
                    and state.get('generation') != args.expected_generation_before):
                raise ValueError('Original generation/epoch changed before replacement')
            if (not state['paused'] or state['is_online'] or state['camera_active']
                    or state.get('ai_controlled') is not False
                    or state['car_id'] != 'bmw_z4_gt3' or state['track_id'] != 'monza'
                    or int(state['window_handle'].removesuffix('ULL')) != window.hwnd):
                raise ValueError('Paused offline AI-off reviewed engine required')
            return state

        before = checked_status(require_original_generation=True)
        axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        if sdk.GetVJDStatus(axis['device_id']) != 1:
            raise ValueError('Released vJoy device required')
        with NativePhysicsReader(timeout=.15) as reader:
            sample = reader.receive()
        if (abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02
                or sample.fields['brake'] <= .98):
            raise ValueError('Native stationary brake hold required')
        write_json(run/'before.json', before)
        write_json(run/'native-before.json', physics_record(sample))
        target = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        if sha256(target) != args.expected_installed_sha256:
            raise ValueError('Installed file changed; inspect before replacement')
        backup = run/'installed-before.lua'
        backup.write_bytes(target.read_bytes())
        if sha256(backup) != args.expected_installed_sha256:
            raise ValueError('Backup hash differs')
        part = target.with_name(target.name+'.flybrain-partial')
        if part.exists():
            raise ValueError('Partial deployment already exists; inspect before replacement')
        source_hash = sha256(source)
        part.write_bytes(source.read_bytes())
        if sha256(part) != source_hash:
            raise ValueError('Source changed while staging deployment')
        # Recheck original PID/generation/idle immediately before the sole replacement.
        checked_status(require_original_generation=True)
        os.replace(part, target)
        result.update(installed=True, new_sha256=source_hash,
                      foreground_before=bool(native.user.GetForegroundWindow() == window.hwnd))
        if sha256(target) != source_hash:
            raise ValueError('Installed hash differs')
        write_json(run/'result.json', result)
        deadline = time.monotonic()+12
        while time.monotonic() < deadline:
            after = checked_status()
            if after['generation'] != before['generation']:
                write_json(run/'after.json', after)
                result.update(status='completed', generation_after=after['generation'],
                              foreground_after=bool(
                                  native.user.GetForegroundWindow() == window.hwnd))
                break
            time.sleep(.1)
        else:
            raise TimeoutError('Installed but reload unobserved; do not repeat replacement')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
