"""Finite stationary MMF stress diagnostic for current four-view GPU transport.

No vehicle command, desktop input, policy, or training data is produced. A failed
group terminates the original lease; it is never acknowledged or silently reused.
"""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=900)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 1000:
        raise ValueError('Finite stationary group budget required')
    axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
    config_path = Path('configs/ac-csp-yaw-retina.json')
    config = json.loads(config_path.read_text(encoding='utf8'))
    status_path = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-session/status.json'
    refuse_leased_transport('stationary MMF transport diagnostic', status_path=status_path)
    windows = WindowsACWindow()
    matches = windows.discover()
    if len(matches) != 1 or windows.user.GetForegroundWindow() == matches[0].hwnd:
        raise ValueError('One original background game window required')
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])

    def idle():
        state, modified = read_published_json(status_path)
        with NativePhysicsReader(timeout=.15) as reader:
            sample = reader.receive()
        if (not 0 <= time.time()-modified <= 2 or state.get('paused') is not True
                or state.get('ai_controlled') is not False or state.get('camera_active')
                or state.get('is_online') or state.get('in_menu')
                or state.get('window_handle') != str(matches[0].hwnd)+'ULL'
                or state.get('car_id') != 'bmw_z4_gt3' or state.get('track_id') != 'monza'
                or sdk.GetVJDStatus(axis['device_id']) != 1
                or abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02
                or sample.fields['brake'] <= .98):
            raise ValueError('Fresh paused stationary full-brake AI-off FREE state required')
        return state

    before = idle()
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'generation': before['generation'], 'frames': args.frames,
        'camera_config_sha256': sha256(config_path), 'role': 'diagnostic; no fit',
        'actor_deployed': False, 'actuator_messages_sent': 0})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'training_admissible': False, 'actor_deployed': False,
              'ui_actions_sent': 0, 'actuator_messages_sent': 0}
    recorder = None
    try:
        retina = CSPCameraRetina(config)
        recorder = CSPVisionRecorder(matches[0].hwnd, config, max_frames=args.frames,
            max_bytes=1024**3, period=.04, image_format='bmp', shared_transport=True,
            batched_shared_copy=True, async_pixel_precompute=True,
            async_frame_persistence=True, mmf_metadata=True, source_timing_v1=True,
            pixel_preprocessor=retina.precompute_pixels)
        while recorder.thread.is_alive():
            recorder.thread.join(timeout=.1)
        result['vision'] = recorder.close()
        if recorder.error is not None or recorder.count != args.frames:
            raise RuntimeError('Original stationary camera lease failed: ' + str(recorder.error))
        after = idle()
        if after['generation'] != before['generation']:
            raise ValueError('Camera generation changed')
        result.update(status='completed_stationary_mmf_transport', frames=recorder.count,
                      session_after=after)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if recorder is not None and recorder.thread.is_alive():
            result['cleanup'] = recorder.close()
        if recorder is not None and not recorder.thread.is_alive() and recorder.triggered:
            result['archive'] = recorder.archive_to(run / 'vision', bundle=True)
        write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)
    if result['status'] != 'completed_stationary_mmf_transport':
        raise RuntimeError(result.get('reason', 'Stationary MMF transport failed'))


if __name__ == '__main__':
    main()
