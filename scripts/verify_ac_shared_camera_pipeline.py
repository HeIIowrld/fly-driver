"""Stationary background camera transport verification; no driving or desktop input."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--paired-reference', action='store_true')
    args = p.parse_args()
    if not 1 <= args.frames <= 600 or args.paired_reference and args.frames > 8:
        raise ValueError('Finite stationary capture budget; reference at most8 frames')
    axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
    config = json.loads(Path('configs/ac-csp-yaw-retina.json').read_text(encoding='utf8'))
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args),
        script_sha256=sha256(__file__),
        capture_source_sha256=sha256('src/flybrain_drive/ac_csp_vision.py')))
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
                  optimizer_updates=0, goal_achieved=False)
    recorder = None
    window_api = WindowsACWindow()
    ai = NativeTeacherStatusReader()
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])

    def observe():
        state, modified = read_published_json(
            Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
        windows = window_api.discover()
        # A paused engine need not advance native packets. Independently read its
        # current snapshot and qualify liveness from the fresh same-engine CSP
        # heartbeat; never claim this static packet is a new physics step.
        with NativePhysicsReader(timeout=.15) as reader:
            sample = reader.receive()
        fields = dict(sample.fields)
        teacher = ai.snapshot()
        if (not 0 <= time.time()-modified <= 2 or not state['paused']
                or state['ai_controlled'] or state['is_online']
                or state['car_id'] != 'bmw_z4_gt3' or state['track_id'] != 'monza'
                or teacher['builtin_ai_controlled'] or abs(fields['speedKmh']) >= .1
                or fields['gas'] >= .02 or fields['brake'] <= .98
                or len(windows) != 1 or sdk.GetVJDStatus(axis['device_id']) != 1
                or window_api.user.GetForegroundWindow() == windows[0].hwnd):
            raise ValueError('Independent stationary background AI-off idle required')
        return dict(session=state, native=fields, native_packet_id=sample.packet_id,
                    paused_native_snapshot_may_have_same_packet=True,
                    engine_pid=windows[0].pid,
                    hwnd=windows[0].hwnd, foreground=False, vjoy_free=True)

    try:
        before = observe()
        write_json(run/'before.json', before)
        if before['session']['camera_active']:
            raise ValueError('Camera must be idle before new owned lease')
        recorder = CSPVisionRecorder(before['hwnd'], config, max_frames=args.frames,
            period=.04, max_bytes=1024**3, image_format='bmp', shared_transport=True,
            verify_png_reference=args.paired_reference)
        while recorder.thread.is_alive():
            recorder.thread.join(timeout=.05)
        result['vision'] = recorder.close()
        after = observe()
        write_json(run/'after.json', after)
        if (after['engine_pid'] != before['engine_pid']
                or after['session']['generation'] != before['session']['generation']
                or recorder.error is not None or recorder.count != args.frames
                or recorder.terminal['status'] != 'completed'):
            raise ValueError('Same complete original camera lease required')
        result['archive'] = recorder.archive_to(run/'vision', bundle=True)
        result.update(status='completed_stationary_shared_pipeline_verification',
            frames=recorder.count, paired_reference_equal=args.paired_reference,
            actor_published_only_after_persistence=True,
            physiological_or_navigation_verification=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if recorder is not None and recorder.thread.is_alive():
            result['cleanup'] = recorder.close()
        if (recorder is not None and not recorder.thread.is_alive()
                and recorder.triggered and 'archive' not in result):
            result['archive'] = recorder.archive_to(run/'vision', bundle=True)
        ai.close()
        write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
