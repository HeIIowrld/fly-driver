"""Read-only stationary SDK/native graphics position agreement; no actuator ownership."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_native_position import NativeDiagnosticPositionReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.calibrate_ac_sdk_steering import validate_stationary_calibration_state
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path('assets/external/ac-native-memory-20261027/source.json')
    upstream = source.parent/'upstream/sim_info.py'
    if sha256(upstream) != json.loads(source.read_text())['sha256']:
        raise ValueError('Pinned source changed')
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), scope=__doc__,
        upstream_sha256=sha256(upstream), actuator_messages_sent=0, ui_actions_sent=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    rows, reader, position, ai = [], None, None, None
    result = dict(status='failed', actuator_messages_sent=0, ui_actions_sent=0,
                  actor_input=False, moving_position_agreement_verified=False)
    try:
        axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine required')
        window = windows[0]
        reader = NativePhysicsReader(timeout=.15, clock=time.perf_counter)
        position, ai = NativeDiagnosticPositionReader(), NativeTeacherStatusReader()
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry',
                                     timeout=.5, clock=time.perf_counter)
        generation = None
        for _ in range(24):
            packet = client.receive()
            flag_before, sample = ai.snapshot(), reader.receive()
            graphics, flag = position.snapshot(), ai.snapshot()
            status, modified = read_published_json(
                Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
            generation = generation or status['generation']
            for flag_value in (flag_before, flag):
                validate_stationary_calibration_state(status, time.time()-modified,
                    flag_value, sample.fields, generation)
            if (sdk.GetVJDStatus(axis['device_id']) != 1 or graphics.status != 'live'
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or api.user.GetForegroundWindow() == window.hwnd
                    or sha256(axis['profile_path']) != axis['profile_sha256']
                    or not packet.raw['producer_session_id'].startswith(
                        'ac-file-'+str(window.pid)+'-')):
                raise ValueError('Fresh background native/SDK identity/free ownership required')
            truth = packet.raw['privileged_teacher']
            if truth['actor_input'] is not False:
                raise ValueError('Position must remain privileged diagnostic-only')
            raw = truth['raw']
            xyz = np.array([raw['world_position_y'], raw['world_position_z'],
                            raw['world_position_x']], float)
            error = float(np.max(abs(xyz-np.array(graphics.coordinates_native_xyz_m))))
            row = dict(native=physics_record(sample), native_graphics=graphics.metadata(),
                sdk=record_packet(packet), sdk_world_native_xyz_m=xyz.tolist(),
                maximum_coordinate_error_m=error, native_ai_before=flag_before,
                native_ai_after=flag, generation=generation, engine_pid=window.pid)
            rows.append(row)
            write_json(run/'samples.json', rows)
            if not np.isfinite(error) or error > .02:
                raise ValueError('Stationary native coordinate agreement exceeds20mm')
            time.sleep(.04)
        result.update(status='completed_stationary_native_position_qualification',
            samples=len(rows), generation=generation, engine_pid=window.pid,
            maximum_coordinate_error_m=max(r['maximum_coordinate_error_m'] for r in rows),
            graphics_prefix_bytes=264, native_coordinates_offset=252,
            producer_session_id=packet.raw['producer_session_id'], vjoy_status=1,
            native_ai=False, foreground=False, samples_sha256=sha256(run/'samples.json'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), samples=len(rows))
    finally:
        for resource in (reader, position, ai):
            if resource is not None:
                resource.close()
        write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
