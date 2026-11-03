"""Bounded paused camera-only stimuli; diagnostic images cannot enter actor streams."""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from PIL import Image

from flybrain_drive.ac_camera_archive import bundle_camera_files
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_clock import CSPClockBounds, initialize_clock
from flybrain_drive.ac_csp_geometry import camera_geometry
from flybrain_drive.ac_csp_vision import copy_shared_group, validate_frame_aid_state
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_shared_texture import SharedTextureReader, verify_shared_frame
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args), plan=plan,
        plan_sha256=sha256(args.plan), script_sha256=sha256(__file__),
        lua_sha256=sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')))
    base = Path(os.environ['LOCALAPPDATA'])/'flybrain'
    api = WindowsACWindow()
    windows = api.discover()
    assert len(windows) == 1
    window = windows[0]
    deployed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    assert sha256(deployed) == sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    root = base/'csp-camera'
    result = dict(status='failed', conditions=[], ui_actions_sent=0,
                  actuator_messages_sent=0, actor_admissible=False, goal_achieved=False)

    def stationary(generation=None):
        state, modified = read_published_json(base/'csp-session/status.json')
        assert 0 <= time.time()-modified <= 2
        assert state['paused'] and not state['ai_controlled'] and not state['is_online']
        assert state['car_id'] == 'bmw_z4_gt3' and state['track_id'] == 'monza'
        assert not state['expert_probe_active'] and not state.get('error')
        assert generation is None or state['generation'] == generation
        assert api.probe(window.hwnd, require_foreground=False) == window
        assert api.user.GetForegroundWindow() != window.hwnd
        with NativePhysicsReader(timeout=.15) as reader:
            sample = reader.receive()
        assert abs(sample.fields['speedKmh']) < .1
        assert sample.fields['gas'] < .02 and sample.fields['brake'] > .98
        return dict(session=state, native=dict(sample.fields), packet_id=sample.packet_id,
                    paused_native_packet_not_claimed_as_new_physics_step=True)

    try:
        for condition in plan['conditions']:
            with SharedTextureReader.reviewed() as gpu:
                output = run/f"condition-{condition['index']:02d}"
                output.mkdir()
                before = stationary()
                assert not before['session']['camera_active']
                write_json(output/'before.json', before)
                generation = before['session']['generation']
                clock = CSPClockBounds(generation,
                    host_quantization=time.get_clock_info('perf_counter').resolution)
                initialize_clock(root, clock, window.hwnd, host_clock=time.perf_counter)
                write_json(output/'clock-initial.json', clock.record())
                rid = 'rotation-'+uuid.uuid4().hex
                directory = root/rid
                request = dict(id=rid, frames=plan['frames'], period_seconds=plan['period_seconds'],
                    deadline_seconds=plan['deadline_seconds'], exposure=.35, image_format='bmp',
                    shared_transport=True, expected_generation=generation,
                    camera_motion_probe=dict(scene=condition['scene'],
                        yaw_left_rad_s=condition['yaw_left_rad_s'],
                        static_pre_seconds=plan['static_pre_seconds'],
                        motion_seconds=plan['motion_seconds']))
                if condition.get('front_mount') is not None:
                    request['camera_motion_probe']['front_mount'] = condition['front_mount']
                write_json(output/'request.json', request)
                write_json(root/'request.json', request)
                rows, outcome = [], dict(status='failed', condition=condition)
                try:
                    deadline = time.perf_counter()+plan['deadline_seconds']+2
                    for index in range(plan['frames']):
                        frame_path = directory/f'frame-{index:04d}.json'
                        while not frame_path.exists():
                            if (directory/'result.json').exists():
                                raise RuntimeError('Producer ended before requested original frame')
                            if time.perf_counter() > deadline:
                                raise TimeoutError('Owned diagnostic camera observation deadline')
                            time.sleep(.002)
                        frame = read(frame_path)
                        assert (frame['request_id'], frame['generation'], frame['index']) == (
                            rid, generation, index)
                        diagnostic = frame['camera_motion_diagnostic']
                        assert diagnostic['actor_admissible'] is False
                        assert diagnostic['physical_body_paused'] is True
                        assert diagnostic['scene'] == condition['scene']
                        assert diagnostic['requested_yaw_left_rad_s'] == condition['yaw_left_rad_s']
                        assert diagnostic['front_mount'] == condition.get(
                            'front_mount', 'native_dashboard_v2')
                        try:
                            validate_frame_aid_state(frame)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError('Actor admission must reject diagnostic frame')
                        camera_geometry(frame['views'])
                        pixels, copies, ack = copy_shared_group(gpu, frame, directory)
                        views = []
                        for view in frame['views']:
                            rgb = pixels[view['view_id']]
                            file = directory/view['file']
                            Image.fromarray(rgb).save(file, format='BMP')
                            views.append(dict(view_id=view['view_id'], file=view['file'],
                                file_sha256=sha256(file),
                                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                render_started_lower=clock.render_start_lower(
                                    view['render_started_clock'], generation)))
                        row = dict(capture_id=index, generation=generation,
                            received=time.perf_counter(), views=views, shared_copy_receipts=copies,
                            file=frame_path.name, file_sha256=sha256(frame_path),
                            camera_motion_diagnostic=diagnostic, actor_admissible=False)
                        verify_shared_frame(frame, row, ack)
                        rows.append(row)
                        write_json(output/'progress.json', dict(original_frames=len(rows)))
                    while not (directory/'result.json').exists():
                        if time.perf_counter() > deadline:
                            raise TimeoutError('No terminal diagnostic producer result')
                        time.sleep(.002)
                    terminal = read(directory/'result.json')
                    assert terminal['status'] == 'completed'
                    assert terminal['frames'] == plan['frames']
                    outcome.update(status='completed', original_frames=len(rows), terminal=terminal)
                finally:
                    if not (directory/'result.json').exists():
                        directory.mkdir(exist_ok=True)
                        (directory/'stop').touch(exist_ok=True)
                        stop_deadline = time.perf_counter()+3
                        while not (directory/'result.json').exists():
                            if time.perf_counter() > stop_deadline:
                                raise RuntimeError('Owned camera terminal unknown; do not retry')
                            time.sleep(.005)
                    write_json(output/'rows.json', rows)
                    outcome['archive'] = bundle_camera_files((directory,), output/'vision')
                    write_json(output/'result.json', outcome)
                after = stationary(generation)
                write_json(output/'after.json', after)
                result['conditions'].append(dict(condition=condition, output=str(output),
                    frames=len(rows), result_sha256=sha256(output/'result.json')))
                write_json(run/'partial-result.json', result)
        result.update(status='completed_paused_camera_rotation_collection',
                      original_frames=sum(c['frames'] for c in result['conditions']))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
