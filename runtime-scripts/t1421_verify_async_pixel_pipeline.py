"""Paused same-shot pixel verification for bounded cache warming; no actor/control use."""
import json
import os
import shutil
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

base = Path(os.environ['LOCALAPPDATA'])/'flybrain'
run = begin_run(base/'runtime-runs/20261029-t1421-async-cache-paired-verification',
    dict(scope=__doc__, program_sha256=sha256(__file__), relevant_tests_passed=38,
         source_snapshot='20261029-t1420', paired_groups=8))
shutil.copyfile(__file__, run/'executed-program.py')
result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False)
recorder = None
try:
    api = WindowsACWindow()
    windows = api.discover()
    assert len(windows) == 1 and windows[0].pid == 36644
    assert api.user.GetForegroundWindow() != windows[0].hwnd
    before, modified = read_published_json(base/'csp-session/status.json')
    assert 0 <= time.time()-modified < 2 and before['paused'] and not before['ai_controlled']
    assert not before['camera_active'] and not before['expert_probe_active'] and not before.get('error')
    with NativePhysicsReader(timeout=.15) as reader:
        native = dict(reader.receive().fields)
    assert abs(native['speedKmh']) < .1 and native['gas'] < .02 and native['brake'] > .98
    write_json(run/'before.json', dict(session=before, native=native))
    config = json.loads(Path('configs/ac-csp-yaw-retina.json').read_text(encoding='utf8'))
    retina = CSPCameraRetina(config)
    recorder = CSPVisionRecorder(windows[0].hwnd, config, max_frames=8, period=.04,
        max_bytes=128*1024**2, image_format='bmp', shared_transport=True,
        verify_png_reference=True, pixel_preprocessor=retina.precompute_pixels,
        async_pixel_precompute=True)
    while recorder.thread.is_alive():
        recorder.thread.join(timeout=.05)
    result['vision'] = recorder.close()
    assert recorder.error is None and recorder.count == 8
    assert recorder.terminal['status'] == 'completed'
    after, modified = read_published_json(base/'csp-session/status.json')
    assert 0 <= time.time()-modified < 2 and after['generation'] == before['generation']
    assert after['paused'] and not after['ai_controlled']
    result.update(status='completed_async_cache_same_shot_verification',
        same_shot_shared_png_pixels_equal=True, groups=8, views_per_group=4,
        source_images_or_clocks_modified=False, actor_qualified=False,
        retinal_temporal_state_unchanged=retina.last_decision is None)
finally:
    if recorder:
        if recorder.thread.is_alive():
            recorder.close()
        if not recorder.thread.is_alive():
            result['archive'] = recorder.archive_to(run/'vision', bundle=True)
    write_json(run/'result.json', result)
print(json.dumps({k:v for k,v in result.items() if k != 'archive'}), flush=True)
