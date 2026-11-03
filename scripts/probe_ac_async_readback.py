"""Paused-only paired PNG/async RGB probe; no actor, lifecycle or actuator commands."""

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.control_ac_session import validate_heartbeat
from scripts.record_ac_wgc import physics_record


def read(path):
    refuse_leased_transport('probe_ac_async_readback')
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    refuse_leased_transport('probe_ac_async_readback')
    native = WindowsACWindow()
    windows = native.discover()
    if len(windows) != 1:
        raise ValueError('Exactly one original engine required')
    window = windows[0]
    session_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'

    def checked_session():
        state = read(session_path)
        validate_heartbeat(state, time.time()-session_path.stat().st_mtime, mutating=True)
        if (not state['paused'] or state['ai_controlled'] or state['camera_active']
                or state['is_online'] or state['car_id'] != 'bmw_z4_gt3'
                or state['track_id'] != 'monza'):
            raise ValueError('Paused offline AI-off camera-idle reviewed car required')
        return state

    before = checked_session()
    refuse_leased_transport('probe_ac_async_readback')
    with NativePhysicsReader(timeout=.15, clock=time.perf_counter) as reader:
        sample = reader.receive()
    if abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02 or (
            sample.fields['brake'] <= .98):
        raise ValueError('Independent native brake-held stationary car required')
    deployed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha256(deployed) != sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
        raise ValueError('Hash-matched deployed camera source required')
    root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-camera'
    request = dict(id='async-probe-'+uuid.uuid4().hex, frames=args.frames,
        period_seconds=.2, deadline_seconds=20, exposure=.35,
        expected_generation=before['generation'], async_readback_probe=True)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862992,
        request=request, deployed_lua_sha256=sha256(deployed), engine_pid=window.pid,
        original_session=before, entry_script_sha256=sha256(__file__)))
    write_json(run/'native-before.json', physics_record(sample))
    directory = root/request['id']
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
                  training_admissible=False, goal_achieved=False)
    try:
        checked_session()
        write_json(root/'request.json', request)
        deadline = time.monotonic()+23
        while not (directory/'result.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Same owned async camera probe has no terminal receipt')
            if native.probe(window.hwnd, require_foreground=False) != window:
                raise ValueError('Original engine changed')
            time.sleep(.02)
        terminal = read(directory/'result.json')
        write_json(run/'producer-result.json', terminal)
        if terminal['status'] != 'completed' or terminal['frames'] != args.frames:
            raise ValueError('Async producer failed: '+str(terminal))
        archive = run/'camera'
        archive.mkdir()
        comparisons = []
        for index in range(args.frames):
            metadata = directory/f'frame-{index:04d}.json'
            frame = read(metadata)
            if (frame['schema'] != 'flybrain.csp.camera.async_probe.frame.v1'
                    or frame['request_id'] != request['id'] or frame['index'] != index
                    or frame['generation'] != before['generation']
                    or {row['view_id'] for row in frame['views']}
                        != {'front', 'left', 'right', 'rear'} or len(frame['views']) != 4):
                raise ValueError('Original probe frame identity differs')
            shutil.copy2(metadata, archive/metadata.name)
            for row in frame['views']:
                files = [row['file'], row['reference_png']]
                if files != [f'{index:04d}-{row["view_id"]}.ppm',
                             f'{index:04d}-{row["view_id"]}-reference.png']:
                    raise ValueError('Probe image paths differ')
                arrays = []
                for name in files:
                    shutil.copy2(directory/name, archive/name)
                    if sha256(directory/name) != sha256(archive/name):
                        raise ValueError('Original image archive differs')
                    with Image.open(archive/name) as image:
                        pixels = np.array(image.convert('RGB'))
                    if pixels.shape != (180, 320, 3):
                        raise ValueError('Probe dimensions differ')
                    arrays.append(pixels)
                error = int(np.abs(arrays[0].astype(int)-arrays[1].astype(int)).max())
                comparisons.append(dict(index=index, view=row['view_id'], maximum_rgb_error=error,
                    async_rgb_sha256=sha256(archive/files[0]),
                    png_sha256=sha256(archive/files[1]),
                    rgb_pixels_sha256=hashlib.sha256(arrays[0].tobytes()).hexdigest(),
                    reference_save_seconds=row['reference_finished_clock']
                        -row['render_call_finished_clock'],
                    callback_delay_seconds=row['async_callback_started_clock']
                        -row['async_requested_clock'],
                    rgb_conversion_seconds=row['conversion_finished_clock']
                        -row['async_callback_started_clock'],
                    rgb_file_seconds=row['async_file_finished_clock']
                        -row['conversion_finished_clock']))
        after = checked_session()
        result.update(status='completed', comparisons=comparisons,
            all_rgb_equal=all(row['maximum_rgb_error'] == 0 for row in comparisons),
            after=after, foreground_after=bool(native.user.GetForegroundWindow() == window.hwnd),
            source_directory=str(directory), source_retained=True,
            paused_paired_probe_only=True, live_transport_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        if directory.exists() and not (directory/'result.json').exists():
            (directory/'stop').touch(exist_ok=True)
    write_json(run/'result.json', result)
    if result['status'] != 'completed' or not result['all_rgb_equal']:
        raise RuntimeError(result.get('reason', 'Async RGB differs from paired PNG'))


if __name__ == '__main__':
    main()
