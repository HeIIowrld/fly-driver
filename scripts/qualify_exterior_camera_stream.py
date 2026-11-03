"""Eight original paused normal-stream frames; camera-only calibration, never fitting data."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_camera_mount import configured_mount, validate_frame_mount
from flybrain_drive.ac_csp_vision import CSPVisionRecorder, validate_frame_aid_state
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_frame_writer import verify_persistence_archive
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.collect_spatial_flight_teacher import idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    refuse_leased_transport('qualify_exterior_camera_stream')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    if configured_mount(config) != 'exterior_front_v3':
        raise ValueError('Explicit new mount qualification required')
    status_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    refuse_leased_transport('qualify_exterior_camera_stream', status_path=status_path)
    generation = json.loads(status_path.read_text(encoding='utf8'))['generation']
    before = idle(generation)
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        sensory_configuration=config, config_sha256=sha256(args.config),
        before=before, role='calibration', fit_allowed=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', role='calibration', fit_allowed=False,
        ui_actions_sent=0, actuator_messages_sent=0, student_deployed=False, goal_achieved=False)
    recorder = None
    try:
        from flybrain_drive.ac_capture import WindowsACWindow

        refuse_leased_transport('qualify_exterior_camera_stream')
        window = WindowsACWindow().discover()[0]
        recorder = CSPVisionRecorder(window.hwnd, config, max_frames=8, period=.08,
            max_bytes=32*1024**2, image_format='bmp', shared_transport=True,
            batched_shared_copy=True, async_frame_persistence=True, mmf_metadata=True)
        recorder.wait_ready()
        deadline = time.monotonic()+10
        while recorder.thread.is_alive():
            if time.monotonic() > deadline:
                raise TimeoutError('Same owned normal-source qualification deadline')
            recorder.thread.join(timeout=.02)
        closed = recorder.close()
        if (closed['error'] is not None or closed['frames'] != 8
                or closed['producer_terminal']['status'] != 'completed'):
            raise ValueError('Entire original paused normal stream did not complete')
        recorder.archive_to(run/'vision', bundle=True)
        with CameraArchive(run/'vision') as archive:
            rows = verify_persistence_archive(archive, expected_mmf=True)
            if len(rows) != 8:
                raise ValueError('Every original source frame required')
            for i in range(8):
                frame = archive.json(f'frame-{i:04d}.json')
                validate_frame_aid_state(frame)
                validate_frame_mount(frame, config)
        # Publication follows cleanup on the next heartbeat; wait only for same generation.
        deadline = time.monotonic()+1
        while True:
            try:
                after = idle(generation)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
        if before['native'] != after['native']:
            raise ValueError('Paused native physical values changed')
        result.update(status='completed_paused_exterior_normal_stream', frames=8,
            closed=closed, after=after, native_values_unchanged=True,
            all_raw_slots_and_persistence_verified=True, moving_stream_qualified=False)
    finally:
        if recorder is not None and recorder.thread.is_alive():
            result['cleanup'] = recorder.close()
        write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
