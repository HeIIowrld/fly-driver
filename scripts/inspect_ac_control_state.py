"""Fresh read-only background/ownership admission evidence; never actuate or use UI input."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.calibrate_ac_sdk_steering import validate_stationary_calibration_state
from scripts.probe_ac_maneuver_plant import native_values
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = begin_run(args.output, dict(scope=__doc__, ui_actions_sent=0,
        actuator_messages_sent=0, program_sha256=sha256(__file__)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0)
    reader = ai = None
    try:
        axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one AC engine required')
        window = windows[0]
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        reader = NativePhysicsReader(timeout=.2, clock=time.perf_counter)
        ai = NativeTeacherStatusReader()
        rows, generation = [], None
        for _ in range(3):
            flag_before = ai.snapshot()
            sample = reader.receive()
            flag_after = ai.snapshot()
            status, modified = read_published_json(
                Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
            age = time.time()-modified
            generation = generation or status['generation']
            foreground = api.user.GetForegroundWindow() == window.hwnd
            vjoy = sdk.GetVJDStatus(axis['device_id'])
            row = dict(native=physics_record(sample), native_ai_before=flag_before,
                native_ai_after=flag_after, csp=status, csp_age_seconds=age,
                vjoy_status=vjoy, foreground=foreground, engine_pid=window.pid,
                native_gear=native_values(sample)['gear'])
            rows.append(row)
            write_json(run/'samples.json', rows)
            for flag in (flag_before, flag_after):
                validate_stationary_calibration_state(status, age, flag, sample.fields, generation)
            if (vjoy != 1 or foreground
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or sha256(axis['profile_path']) != axis['profile_sha256']):
                raise ValueError('Fresh background/free ownership/profile admission failed')
            time.sleep(.04)
        result.update(status='completed_fresh_read_only_admission', engine_pid=window.pid,
            generation=generation, vjoy_status=vjoy, foreground=foreground,
            native_ai=False, native_gear=rows[-1]['native_gear'],
            native_speed_kmh=sample.fields['speedKmh'], native_gas=sample.fields['gas'],
            native_brake=sample.fields['brake'], samples=len(rows), csp_age_seconds=age)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if reader is not None:
            reader.close()
        if ai is not None:
            ai.close()
        write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
