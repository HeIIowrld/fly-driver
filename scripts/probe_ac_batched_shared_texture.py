"""Same immutable paused frame: compare sequential and batch GPU readback with PNG."""

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
    assert len(windows) == 1
    window = windows[0]
    local = Path(os.environ['LOCALAPPDATA'])/'flybrain'
    before, modified, _ = read_session_heartbeat(local/'csp-session/status.json')
    validate_heartbeat(before, time.time()-modified, mutating=True)
    assert before['paused'] and not before['ai_controlled'] and not before['camera_active']
    assert not before['expert_probe_active'] and not before['is_online']
    with NativePhysicsReader(timeout=.15) as reader:
        sample = reader.receive()
    assert abs(sample.fields['speedKmh']) < .1 and sample.fields['gas'] < .02
    assert sample.fields['brake'] > .98
    deployed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    assert sha256(deployed) == sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    root = local/'csp-camera'
    request = dict(id='batch-probe-'+uuid.uuid4().hex, frames=1, period_seconds=.2,
        deadline_seconds=20, exposure=.35, expected_generation=before['generation'],
        shared_texture_probe=True)
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        reader_source_sha256=sha256('src/flybrain_drive/ac_shared_texture.py'),
        before=before, request=request, engine_pid=window.pid, hwnd=window.hwnd,
        order='One warmup pair then four alternating legacy/batch pairs on same immutable source',
        ui_actions_sent=0, actuator_messages_sent=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'native-before.json', physics_record(sample))
    directory = root/request['id']
    result = dict(status='failed', goal_achieved=False)
    try:
        with SharedTextureReader.reviewed() as gpu:
            write_json(root/'request.json', request)
            deadline = time.monotonic()+15
            while not (directory/'frame-0000.json').exists():
                assert time.monotonic() < deadline, 'No immutable same-engine frame'
                assert api.probe(window.hwnd, require_foreground=False) == window
                time.sleep(.005)
            frame = read(directory/'frame-0000.json')
            assert frame['request_id'] == request['id']
            assert frame['generation'] == before['generation']
            shutil.copy2(directory/'frame-0000.json', run/'frame-0000.json')
            views = frame['views']
            handles = [v['shared_texture_handle'] for v in views]
            references = []
            for v in views:
                src = directory/v['file']
                shutil.copy2(src, run/v['file'])
                references.append(np.array(Image.open(src).convert('RGB')))
            rows = []
            for repeat in range(-1, 4):
                order = ('legacy', 'batch') if repeat % 2 else ('batch', 'legacy')
                for method in order:
                    started = time.perf_counter()
                    copies = (gpu.read_group(handles) if method == 'batch' else
                              [gpu.read(handle) for handle in handles])
                    elapsed = time.perf_counter()-started
                    errors = [int(np.abs(rgb.astype(int)-ref.astype(int)).max())
                              for (rgb, _), ref in zip(copies, references, strict=True)]
                    assert max(errors) == 0, 'GPU path differs from original same-shot PNG'
                    rows.append(dict(repeat=repeat, warmup=repeat == -1, method=method,
                        group_seconds=elapsed, maximum_rgb_errors=errors,
                        copies=[dict(receipt, view_id=v['view_id'],
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                            for v, (rgb, receipt) in zip(views, copies, strict=True)]))
                write_json(run/'progress.json', rows)
            measured = {method: [r['group_seconds'] for r in rows
                         if not r['warmup'] and r['method'] == method]
                         for method in ('legacy', 'batch')}
            result.update(status='completed_same_immutable_readback_comparison', comparisons=rows,
                measured_group_seconds=measured, all_rgb_equal=True, adapter=gpu.adapter,
                feature_level=gpu.feature_level, source_texture_modified=False,
                paused_only_not_live_latency_qualification=True)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if directory.exists():
            (directory/'stop').touch(exist_ok=True)
            deadline = time.monotonic()+3
            while not (directory/'result.json').exists() and time.monotonic() < deadline:
                time.sleep(.01)
        result['owned_camera_terminal'] = read(directory/'result.json') if (
            directory/'result.json').exists() else None
        write_json(run/'result.json', result)
    assert result['status'] == 'completed_same_immutable_readback_comparison', result.get('reason')
    assert result['owned_camera_terminal'] is not None


if __name__ == '__main__':
    main()
