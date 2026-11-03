"""Read-only live teacher inference; no vJoy, keyboard, reset or actor action interface."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader, decode_static_identity_prefix
from flybrain_drive.ac_sac_teacher import PrivilegedSACTeacher, TeacherReferenceCurve
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.ac_teacher_observation import (
    PrivilegedTeacherObservationBuilder,
    TeacherInputUnavailable,
)
from flybrain_drive.evidence import begin_run, write_json
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 10000:
        raise ValueError('Finite read-only sample budget required')
    torch.set_num_threads(1)
    teacher_config = json.loads(Path('configs/ac-sac-replay.json').read_text())
    geometry_config = json.loads(Path('configs/ac-teacher-geometry-result.json').read_text())[
        'geometry_configuration']
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args),
        teacher_configuration=teacher_config, geometry_configuration=geometry_config,
        nominal_period_seconds=.04, actual_time_mapping_verified=False,
        actuator_messages_sent=0))
    count, inferred = 0, 0
    try:
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        if identity['car_model'] != 'bmw_z4_gt3' or identity['track'] != 'monza':
            raise ValueError('Live car/track differs from reviewed teacher')
        write_json(run / 'identity.json', identity)
        window_api = WindowsACWindow()
        windows = window_api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine window required')
        window = windows[0]
        geometry = PrivilegedTrackGeometry(geometry_config['track_csv'],
            geometry_config['reference_csv'], track_sha256=geometry_config['track_sha256'],
            reference_sha256=geometry_config['reference_sha256'])
        curve = TeacherReferenceCurve(geometry_config['reference_csv'])
        builder = PrivilegedTeacherObservationBuilder(geometry, curve,
            profile=teacher_config['profile'], minimum_interval=.015, maximum_interval=.12)
        teacher = PrivilegedSACTeacher(teacher_config['checkpoint'],
                                     teacher_config['checkpoint_sha256'])
        directory = Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry'
        client = FileTelemetryClient(directory, timeout=.5)
        with NativePhysicsReader(timeout=.5) as native:
            with (run / 'privileged-shadow.jsonl').open('x', encoding='utf8') as stream:
                for _ in range(args.samples):
                    started = time.monotonic()
                    if window_api.probe(window.hwnd, require_foreground=False) != window:
                        raise ValueError('Original engine/window changed during shadow collection')
                    packet = client.receive()
                    physics = native.receive()
                    result = builder.build(packet, physics)
                    delta = None
                    if result.observation is not None:
                        delta = teacher.predict(result.observation[None])[
                            'deterministic_policy_action'][0].tolist()
                        inferred += 1
                    row = dict(sdk=record_packet(packet), native=physics_record(physics),
                        producer_session_id=result.producer_session_id,
                        host_receipt_skew_seconds=result.host_receipt_skew_seconds,
                        observation=result.observation.tolist()
                            if result.observation is not None else None,
                        teacher_policy_delta_request=delta, request_applied=False,
                        was_foreground_at_record=bool(
                            window_api.user.GetForegroundWindow() == window.hwnd),
                        actor_stream_created=False)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    count += 1
                    time.sleep(max(0, .04 - (time.monotonic() - started)))
        write_json(run / 'result.json', dict(status='completed', samples=count,
            teacher_inferences=inferred, requests_applied=0, actuator_messages_sent=0,
            time_mapping_verified=False, teacher_driving_verified=False, goal_achieved=False))
    except TeacherInputUnavailable as error:
        write_json(run / 'result.json', dict(status='not_ready', samples=count,
            teacher_inferences=inferred, reason=str(error), actuator_messages_sent=0,
            teacher_driving_verified=False, goal_achieved=False))
    except Exception as error:
        write_json(run / 'result.json', dict(status='failed', samples=count,
            teacher_inferences=inferred, error=type(error).__name__, message=str(error),
            actuator_messages_sent=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
