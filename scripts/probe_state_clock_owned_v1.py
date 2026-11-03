"""Finite read-only CSP engine camera probe and hash-verified local collection."""

import argparse
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--period', type=float, default=.2)
    parser.add_argument('--exposure', type=float, default=.35)
    parser.add_argument('--allow-ideal-line-probe', action='store_true',
                        help='At most8 paused stationary frames; never training-admissible')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.allow_ideal_line_probe and args.frames > 8:
        raise ValueError('Ideal-line diagnostic is limited to8paused frames')
    if not 1 <= args.frames <= 100 or not np.isfinite(args.period) or not .04 <= args.period <= 2:
        raise ValueError('Finite frame and period budgets required')
    if not np.isfinite(args.exposure) or not .01 <= args.exposure <= 4:
        raise ValueError('Explicit bounded camera exposure required')
    timeout = min(60., max(10., args.frames * args.period + 5.))
    root = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-camera'
    status_path = root / 'status.json'
    if not status_path.exists() or time.time() - status_path.stat().st_mtime > 3:
        raise ValueError('Live camera app heartbeat required')
    status = json.loads(status_path.read_text())
    if status.get('patch_version') != '0.2.11' or status.get('state') == 'capturing':
        raise ValueError('Reviewed CSP version and idle camera app required')
    native = WindowsACWindow()
    windows = native.discover()
    if len(windows) != 1:
        raise ValueError('Exactly one original AC engine required')
    window = windows[0]
    foreground = bool(native.user.GetForegroundWindow() == window.hwnd)
    deployed = Path(window.executable).parent / 'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    source = Path('integrations/ac_csp/candidates/FlyBrainCamera-state-clock-owned-v1.lua')
    if sha256(source) != sha256(deployed):
        raise ValueError('Deployed camera app differs from reviewed source')
    request_id = 'probe-' + uuid.uuid4().hex
    request = dict(id=request_id, frames=args.frames, period_seconds=args.period,
                   allow_ideal_line_probe=args.allow_ideal_line_probe,
                   deadline_seconds=timeout, exposure=args.exposure)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862803,
        request=request, engine_pid=window.pid, hwnd=window.hwnd, before=status,
        was_foreground_at_request=foreground, deployed_lua_sha256=sha256(deployed),
        ui_actions_sent=0, actuator_messages_sent=0))
    shutil.copy2(deployed, run / 'deployed-camera.lua')
    if sha256(run / 'deployed-camera.lua') != sha256(deployed):
        raise ValueError('Camera source snapshot differs')
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
                  actor_stream_created=False, training_admissible=False, goal_achieved=False)
    try:
        write_json(root / 'request.json', request)
        deadline = time.monotonic() + timeout + 2
        directory = root / request_id
        while not (directory / 'result.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('No terminal camera result within request deadline')
            if native.probe(window.hwnd, require_foreground=False) != window:
                raise ValueError('Original engine changed while capturing')
            time.sleep(.1)
        raw_result = json.loads((directory / 'result.json').read_text())
        write_json(run / 'engine-result.json', raw_result)
        if raw_result.get('status') != 'completed' or raw_result.get('frames') != args.frames:
            raise ValueError('Engine capture failed: ' + str(raw_result))
        archive = run / 'camera'
        archive.mkdir()
        manifest, frames, render_ids = [], [], []
        for i in range(args.frames):
            frame_path = directory / f'frame-{i:04d}.json'
            frame = json.loads(frame_path.read_text())
            if (frame['schema'] != 'flybrain.csp.camera.frame.v1'
                    or frame['request_id'] != request_id or frame['index'] != i
                    or {v['view_id'] for v in frame['views']} != {'front', 'left', 'right', 'rear'}
                    or len(frame['views']) != 4):
                raise ValueError('Camera frame identity or view coverage changed')
            render_id = int(re.sub(r'ULL$', '', str(frame['render_frame'])))
            render_ids.append(render_id)
            checks = []
            for view in frame['views']:
                filename = f'{i:04d}-{view["view_id"]}.png'
                if view['file'] != filename:
                    raise ValueError('Camera file path changed')
                src, dst = directory / filename, archive / filename
                shutil.copy2(src, dst)
                if sha256(src) != sha256(dst):
                    raise ValueError('Image archive copy differs')
                with Image.open(dst) as image:
                    rgb = np.array(image.convert('RGB'))
                if rgb.shape != (180, 320, 3):
                    raise ValueError('Camera resolution differs')
                checks.append(dict(view_id=view['view_id'], file=filename, sha256=sha256(dst),
                    rgb_mean=rgb.mean((0, 1)).tolist(), rgb_std=float(rgb.std()),
                    colors=len(np.unique(rgb.reshape(-1, 3), axis=0)),
                    render_and_save_seconds=view['save_finished_clock']-view['render_started_clock']))
                manifest.append(dict(file='camera/' + filename, sha256=sha256(dst)))
            shutil.copy2(frame_path, archive / frame_path.name)
            manifest.append(dict(file='camera/' + frame_path.name,
                                 sha256=sha256(archive / frame_path.name)))
            frames.append(dict(index=i, render_frame=render_id, views=checks))
        if any(b <= a for a, b in zip(render_ids[:-1], render_ids[1:], strict=True)):
            raise ValueError('Engine render frame did not advance between capture groups')
        write_json(run / 'archive.json', manifest)
        result.update(status='completed', request_id=request_id, frames=frames,
            source_directory=str(directory), source_retained=True, all_hashes_verified=True,
            render_frame_ids_increased=True,
            all_images_nonconstant=all(v['colors'] > 16 for f in frames for v in f['views']),
            was_foreground_at_request=foreground,
            was_foreground_at_finish=bool(native.user.GetForegroundWindow() == window.hwnd),
            all_four_views_same_exposure_verified=False,
            camera_axis_and_optics_independently_calibrated=False,
            shared_texture_consumer_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        write_json(run / 'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result.get('reason', 'Camera probe failed'))


if __name__ == '__main__':
    main()
