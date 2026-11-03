"""Read-only paused collection baseline and privileged route; never actuator admission."""

import argparse
import json
import math
import os
import struct
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_clock import GRAPHICS_PREFIX, decode_graphics_timing
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_native_position import decode_diagnostic_position
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--axis-config', default='configs/ac-axis-probe.json')
    args = parser.parse_args()
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), scope=__doc__,
        axis_config=args.axis_config, axis_config_sha256=sha256(args.axis_config),
        ui_actions_sent=0, actuator_messages_sent=0, actor_input=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', controller_admission=False, actor_input=False,
        ui_actions_sent=0, actuator_messages_sent=0)
    ai = None
    try:
        state, modified = read_published_json(
            Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('One original engine required')
        window = windows[0]
        axis = json.loads(Path(args.axis_config).read_text(encoding='utf8'))
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        ai = NativeTeacherStatusReader()
        with NativePhysicsReader(timeout=.15, clock=time.perf_counter) as physics:
            sample = physics.receive()
        flag = ai.snapshot()
        with ExistingNamedMapping('Local\\acpmf_graphics', 264) as mapping:
            raw, second = mapping.read(), mapping.read()
        if raw != second:
            raise ValueError('Paused graphics baseline changed while reading')
        now = time.perf_counter()
        position = decode_diagnostic_position(raw, now)
        timing = decode_graphics_timing(raw[:GRAPHICS_PREFIX.size], now)
        spline = struct.unpack_from('<f', raw, 248)[0]
        age = time.time()-modified
        write_json(run/'raw-baselines.json', dict(native=physics_record(sample), native_ai=flag,
            native_graphics=position.metadata(), csp=state, csp_age_seconds=age))
        foreground = api.user.GetForegroundWindow() == window.hwnd
        if (not 0 <= age <= 2 or state.get('paused') is not True
                or state.get('ai_controlled') is not False or state.get('camera_active')
                or state.get('expert_probe_active') or state.get('pending_receipt_id')
                or state.get('error') or state.get('is_online') or state.get('in_menu')
                or position.status != 'pause' or sdk.GetVJDStatus(axis['device_id']) != 1
                or foreground or api.probe(window.hwnd, require_foreground=False) != window
                or sha256(axis['profile_path']) != axis['profile_sha256']
                or not math.isfinite(spline) or not 0 <= spline < 1
                or abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02
                or sample.fields['brake'] <= .98):
            raise ValueError('Paused same-engine/full-brake/background/FREE baseline required')
        result.update(status='completed_read_only_paused_collection_baseline',
            engine_pid=window.pid, generation=state['generation'], vjoy_status=1,
            native_ai_frozen_baseline=flag['builtin_ai_controlled'],
            native_speed_kmh=sample.fields['speedKmh'], native_gas=sample.fields['gas'],
            native_brake=sample.fields['brake'], foreground=False, csp_age_seconds=age,
            diagnostic_route=dict(spline_position=spline, lap_count=timing.completed_laps,
                actor_input=False, source='paused native graphics baseline, not fresh motion'),
            new_physics_advance_verified=False,
            meaning='Planning/baseline only; actual API lifecycle guards must run again.')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if ai is not None:
            ai.close()
        write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
