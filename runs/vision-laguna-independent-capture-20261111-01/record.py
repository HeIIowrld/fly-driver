"""Capture a bounded offline Laguna Seca stream with the reviewed CSP recorder."""
from pathlib import Path
import os,time

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_published_state import read_published_json
from scripts.control_ac_session import validate_heartbeat
from train_vision_teacher_line import digest,read,write

out=Path(__file__).parent
config=Path("configs/ac-csp-exterior-front-v3-retina.json")
protocol=Path("runs/vision-corner-independent-track-options-20261111-01/ks_laguna_seca-protocol.json")
status=Path(os.environ["LOCALAPPDATA"])/"flybrain/csp-session/status.json"
source="f4df5e21e3c6acbf346d7bacf52f63eaf049b50d9d5b3cf353e7d76d390bcb4a"
before,modified=read_published_json(status)
validate_heartbeat(before,time.time()-modified,mutating=False)
windows=WindowsACWindow().discover()
assert len(windows)==1 and windows[0].pid==21756
assert int(before["window_handle"].removesuffix("ULL"))==windows[0].hwnd
assert (before["track_id"]=="ks_laguna_seca" and before["car_id"]=="bmw_z4_gt3" and before["paused"]
        and before["brake"]>.98 and abs(before["speed_kmh"])<.1 and not before["is_online"]
        and not before["ai_controlled"] and not before["camera_active"] and not before["ideal_line_shown"])
write(out/"manifest.json",dict(revision=read(protocol)["revision"],seed=None,frames=1000,period_s=.1,
    hashes={str(p):digest(p) for p in (config,protocol,Path(__file__))},source_sha256=source,
    role="independent_evaluation",before=before,actor_control_sent=False))
recorder=None
result=dict(status="failed")
try:
    recorder=CSPVisionRecorder(windows[0].hwnd,read(config),max_frames=1000,period=.1,
        max_bytes=2*1024**3,image_format="bmp",shared_transport=True,batched_shared_copy=True,
        async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,
        expected_source_sha256=source,expected_track="ks_laguna_seca",source_body_recording=True)
    recorder.wait_ready(timeout=8.)
    write(out/"ready.json",dict(request_id=recorder.request_id,engine_pid=windows[0].pid))
    print("CSP capture ready",flush=True)
    while recorder.thread.is_alive() and not (out/"stop").exists():
        recorder.thread.join(.25)
    if recorder.thread.is_alive():
        recorder.request_stop()
    closed=recorder.close()
    write(out/"capture-ended.json",dict(frames=recorder.count,closed=closed))
    print("Capture ended",recorder.count,flush=True)
    archive=recorder.archive_to(out/"vision",bundle=True)
    result.update(status="captured" if recorder.count else "empty_capture",
                  frames=recorder.count,archive=archive,closed=closed)
except Exception as e:
    result["reason"]=repr(e)
    raise
finally:
    if recorder is not None and recorder.thread.is_alive():
        result["cleanup"]=recorder.close()
    write(out/"result.json",dict(result,termination="owned_capture_end"))
