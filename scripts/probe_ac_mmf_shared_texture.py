"""Paused eight-group MMF metadata/ACK qualification against same-shot producer PNG."""

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_geometry import camera_geometry
from flybrain_drive.ac_csp_vision import copy_shared_group, validate_frame_aid_state
from flybrain_drive.ac_metadata_mmf import MetadataMMF
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_shared_texture import SharedTextureReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat
from scripts.record_ac_wgc import physics_record


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    api = WindowsACWindow()
    windows = api.discover()
    if len(windows) != 1:
        raise ValueError('Exactly one reviewed original engine required')
    window = windows[0]
    local = Path(os.environ['LOCALAPPDATA'])/'flybrain'
    root = local/'csp-camera'
    before, modified, _ = read_session_heartbeat(local/'csp-session/status.json')
    validate_heartbeat(before, time.time()-modified, mutating=True)
    if (not before['paused'] or before['ai_controlled'] or before['camera_active']
            or before['expert_probe_active'] or before['is_online']
            or before['ideal_line_shown'] is not False):
        raise ValueError('Paused offline aid-OFF and idle CSP teacher required')
    with NativePhysicsReader(timeout=.15) as reader:
        native = reader.receive()
    if (abs(native.fields['speedKmh']) >= .1 or native.fields['gas'] >= .02
            or native.fields['brake'] <= .98):
        raise ValueError('Independent native stationary full-brake state required')
    axis = read('configs/ac-axis-probe.json')
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    if sdk.GetVJDStatus(axis['device_id']) != 1:
        raise ValueError('Released controller required')
    deployed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha256(deployed) != sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
        raise ValueError('Deployed producer differs from reviewed source')
    request = dict(id='mmf-probe-'+uuid.uuid4().hex, frames=8, period_seconds=.08,
        deadline_seconds=10, exposure=.35, expected_generation=before['generation'],
        shared_transport=True, image_format='bmp', mmf_metadata=True, verify_png_reference=True)
    directory = root/request['id']
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, request=request,
        program_sha256=sha256(__file__), installed_lua_sha256=sha256(deployed),
        native_before=physics_record(native), session_before=before,
        role='calibration', fit_allowed=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    shutil.copy2(deployed, run/'executed-producer.lua')
    result = dict(status='failed', goal_achieved=False, student_deployed=False,
                  ui_actions_sent=0, vjoy_messages_sent=0, teacher_api_calls=0,
                  paused_only=True, live_stream_qualified=False, role='calibration')
    channel, triggered = None, False
    retained, pixels = [], {}
    try:
        channel = MetadataMMF(request['id'], before['generation'])
        with SharedTextureReader.reviewed() as gpu:
            write_json(root/'request.json', request)
            triggered = True
            deadline = time.monotonic()+12
            for index in range(8):
                while True:
                    received = channel.poll(index)
                    if received is not None:
                        break
                    terminal = directory/'result.json'
                    if terminal.exists():
                        raise ValueError('Producer ended before expected MMF frame: '
                                         + str(read(terminal)))
                    if time.monotonic() >= deadline:
                        raise TimeoutError('MMF original request deadline exceeded')
                    if api.probe(window.hwnd, require_foreground=False) != window:
                        raise ValueError('Original engine changed')
                    time.sleep(.001)
                frame, raw, receipt = received
                item = dict(index=index, raw=raw, receipt=receipt, frame=frame,
                            stage='metadata_received')
                retained.append(item)
                validate_frame_aid_state(frame)
                camera_geometry(frame['views'])
                if (directory/f'frame-{index:04d}.json').exists():
                    raise ValueError('MMF producer unexpectedly wrote frame metadata to disk')

                def acknowledge(ack, item=item):
                    item['ack_payload'] = channel.acknowledge(ack)

                rgb, copied, ack = copy_shared_group(gpu, frame, directory,
                    batched=True, ack_writer=acknowledge)
                pixels[index] = rgb
                item.update(copied=copied, ack=ack, stage='copied_acknowledged')
                if (directory/f'ack-{index:04d}.json').exists():
                    raise ValueError('MMF consumer unexpectedly wrote ACK to disk')
                reference = directory/f'reference-{index:04d}.json'
                while not reference.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Same-shot producer PNG reference missing')
                    time.sleep(.001)
                confirmation = read(reference)
                if (confirmation['request_id'] != request['id']
                        or confirmation['generation'] != before['generation']
                        or confirmation['capture_id'] != index
                        or confirmation['stage'] != 'after_host_copy'):
                    raise ValueError('Same-shot PNG identity differs')
                errors = {}
                for view in frame['views']:
                    name = view['view_id']
                    with Image.open(directory/f'{index:04d}-{name}-reference.png') as image:
                        original = np.asarray(image.convert('RGB'))
                    errors[name] = int(np.abs(original.astype(int)-rgb[name].astype(int)).max())
                    if errors[name] != 0:
                        raise ValueError('MMF shared pixels differ from producer same-shot PNG')
                item.update(png_confirmation=confirmation, maximum_rgb_errors=errors,
                            stage='reference_verified')
            while not (directory/'result.json').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Original MMF capture terminal missing')
                time.sleep(.005)
            terminal = read(directory/'result.json')
            if terminal['status'] != 'completed' or terminal['frames'] != 8:
                raise ValueError('Original MMF capture did not complete')
        result.update(status='completed_paused_mmf_same_shot_qualification', groups=8,
                      pixel_comparisons=32, maximum_rgb_error=0,
                      producer_metadata_files=len(list(directory.glob('frame-*.json'))),
                      consumer_ack_files=len(list(directory.glob('ack-*.json'))),
                      source_texture_modified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if triggered and not (directory/'result.json').exists():
            (directory/'stop').touch(exist_ok=True)
        if triggered:
            deadline = time.monotonic()+3
            while not (directory/'result.json').exists() and time.monotonic() < deadline:
                time.sleep(.01)
        result['producer_terminal'] = read(directory/'result.json') if (
            directory/'result.json').exists() else None
        if channel:
            channel.close()
        # Preserve original received bytes after capture, including failed partial histories.
        history = []
        for item in retained:
            index = item['index']
            (run/f'frame-{index:04d}.json').write_bytes(item.pop('raw'))
            if 'ack_payload' in item:
                (run/f'ack-{index:04d}.json').write_bytes(item.pop('ack_payload'))
            for name, value in pixels.get(index, {}).items():
                np.save(run/f'{index:04d}-{name}-rgb.npy', value, allow_pickle=False)
                reference = directory/f'{index:04d}-{name}-reference.png'
                if reference.exists():
                    shutil.copy2(reference, run/reference.name)
            history.append(item)
        write_json(run/'history.json', history)
        deadline = time.monotonic()+2
        while True:
            after, modified, _ = read_session_heartbeat(local/'csp-session/status.json')
            if not after['camera_active'] or time.monotonic() > deadline:
                break
            time.sleep(.01)
        result['session_after'] = after
        result['foreground_after'] = bool(api.user.GetForegroundWindow() == window.hwnd)
        if (time.time()-modified > 2 or not after['paused'] or after['camera_active']
                or after['expert_probe_active'] or after['generation'] != before['generation']):
            result.update(status='failed', reason='Known original paused terminal required')
        with NativePhysicsReader(timeout=.15) as reader:
            native_after = reader.receive()
            result['native_after'] = physics_record(native_after)
            if (abs(native_after.fields['speedKmh']) >= .1 or native_after.fields['gas'] >= .02
                    or native_after.fields['brake'] <= .98):
                result.update(status='failed', reason='Native stationary brake hold changed')
        write_json(run/'result.json', result)
    if result['status'] != 'completed_paused_mmf_same_shot_qualification':
        raise RuntimeError(result.get('reason', 'MMF qualification failed'))
    print(json.dumps({k:result[k] for k in ('status', 'groups', 'pixel_comparisons',
        'maximum_rgb_error', 'producer_metadata_files', 'consumer_ack_files')}), flush=True)


if __name__ == '__main__':
    main()
