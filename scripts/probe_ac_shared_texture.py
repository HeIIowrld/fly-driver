"""Paused-only direct shared-texture/PNG comparison; source held immutable by owned Lua lease."""

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
from flybrain_drive.ac_shared_texture import SharedTextureReader
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.control_ac_session import validate_heartbeat
from scripts.record_ac_wgc import physics_record


def read(path):
    refuse_leased_transport('probe_ac_shared_texture')
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapter-index', type=int, choices=range(8), default=1)
    args = parser.parse_args()
    refuse_leased_transport('probe_ac_shared_texture')
    native = WindowsACWindow()
    windows = native.discover()
    if len(windows) != 1:
        raise ValueError('Exactly one original engine required')
    window = windows[0]
    session_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    before = read(session_path)
    validate_heartbeat(before, time.time()-session_path.stat().st_mtime, mutating=True)
    if (not before['paused'] or before['ai_controlled'] or before['camera_active']
            or before['is_online'] or before['car_id'] != 'bmw_z4_gt3'
            or before['track_id'] != 'monza'):
        raise ValueError('Paused offline AI-off camera-idle reviewed car required')
    refuse_leased_transport('probe_ac_shared_texture')
    with NativePhysicsReader(timeout=.15, clock=time.perf_counter) as reader:
        sample = reader.receive()
    if abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02 or (
            sample.fields['brake'] <= .98):
        raise ValueError('Independent stationary native brake hold required')
    source = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha256(source) != sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
        raise ValueError('Reviewed deployed source hash required')
    root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-camera'
    request = dict(id='shared-probe-'+uuid.uuid4().hex, frames=1, period_seconds=.2,
        deadline_seconds=20, exposure=.35, expected_generation=before['generation'],
        shared_texture_probe=True)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862992,
        request=request, deployed_lua_sha256=sha256(source), engine_pid=window.pid,
        original_session=before, entry_script_sha256=sha256(__file__),
        abi_source_sha256=sha256('assets/external/d3d11-shared-texture-api-20261028/d3d11.h')))
    write_json(run/'native-before.json', physics_record(sample))
    directory, result = root/request['id'], dict(status='failed', ui_actions_sent=0,
        actuator_messages_sent=0, training_admissible=False, goal_achieved=False)
    try:
        refuse_leased_transport('probe_ac_shared_texture')
        with SharedTextureReader(adapter_index=args.adapter_index) as gpu:
            write_json(root/'request.json', request)
            deadline = time.monotonic()+15
            while not (directory/'frame-0000.json').exists():
                if (directory/'result.json').exists():
                    raise RuntimeError('Producer ended before stable shared frame: '+str(
                        read(directory/'result.json')))
                if time.monotonic() > deadline:
                    raise TimeoutError('No frame in the same owned shared lease')
                if native.probe(window.hwnd, require_foreground=False) != window:
                    raise ValueError('Original engine changed')
                time.sleep(.005)
            frame_file = directory/'frame-0000.json'
            frame = read(frame_file)
            if (frame['generation'] != before['generation'] or frame['request_id'] != request['id']
                    or frame['index'] != 0 or len(frame['views']) != 4):
                raise ValueError('Original shared camera identity differs')
            shutil.copy2(frame_file, run/'frame-0000.json')
            rows = []
            for view in frame['views']:
                if view['file'] != f'0000-{view["view_id"]}.png':
                    raise ValueError('Pinned paired PNG filename required')
                shutil.copy2(directory/view['file'], run/view['file'])
                with Image.open(run/view['file']) as image:
                    reference = np.array(image.convert('RGB'))
                for repeat in range(4):
                    pixels, receipt = gpu.read(view['shared_texture_handle'])
                    path = run/f'{view["view_id"]}-{repeat:02d}.npy'
                    np.save(path, pixels, allow_pickle=False)
                    error = int(np.abs(pixels.astype(int)-reference.astype(int)).max())
                    rows.append(dict(view=view['view_id'], repeat=repeat,
                        maximum_rgb_error=error, read_seconds=receipt['finished_at']
                            -receipt['started_at'], receipt=receipt,
                        rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                        numpy_sha256=sha256(path), paired_png_sha256=sha256(run/view['file'])))
                    write_json(run/'progress.json', rows)
            result.update(status='completed', comparisons=rows,
                all_rgb_equal=all(row['maximum_rgb_error'] == 0 for row in rows),
                feature_level=gpu.feature_level, adapter=gpu.adapter, source_texture_modified=False,
                paused_immutable_probe_only=True, live_synchronization_verified=False,
                foreground_after=bool(native.user.GetForegroundWindow() == window.hwnd))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if directory.exists() and not (directory/'result.json').exists():
            (directory/'stop').touch(exist_ok=True)
            deadline = time.monotonic()+3
            while not (directory/'result.json').exists() and time.monotonic() < deadline:
                time.sleep(.01)
        if (directory/'result.json').exists():
            terminal = read(directory/'result.json')
            write_json(run/'producer-result.json', terminal)
            result['owned_camera_terminated'] = terminal['status'] == 'stopped'
        else:
            result['owned_camera_terminated'] = False
        result.update(source_directory=str(directory), source_retained=True)
        write_json(run/'result.json', result)
    if result['status'] != 'completed' or not result['all_rgb_equal'] or (
            not result['owned_camera_terminated']):
        raise RuntimeError(result.get('reason', 'Shared RGB/cleanup verification failed'))


if __name__ == '__main__':
    main()
