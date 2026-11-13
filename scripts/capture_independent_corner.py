"""Capture a bounded four-view stream for a frozen independent corner evaluation."""
import argparse
import os
from pathlib import Path
import time

from evaluate_fused_corner import load_frozen
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_published_state import read_published_json
from scripts.control_ac_session import validate_heartbeat
from train_vision_teacher_line import digest, read, write


def check_session(track, expected_source):
    status = Path(os.environ["LOCALAPPDATA"])/"flybrain/csp-session/status.json"
    before, modified = read_published_json(status)
    validate_heartbeat(before, time.time()-modified, mutating=False)
    windows = WindowsACWindow().discover()
    if len(windows) != 1 or int(before["window_handle"].removesuffix("ULL")) != windows[0].hwnd:
        raise ValueError("Expected exactly one original AC engine matching the CSP heartbeat")
    window = windows[0]
    source = Path(window.executable).parent/"apps/lua/FlyBrainCamera/FlyBrainCamera.lua"
    actual = digest(source)
    requirements = {
        "track": before["track_id"] == track,
        "car": before["car_id"] == "bmw_z4_gt3",
        "offline": not before["is_online"],
        "paused": before["paused"],
        "stationary_braked": abs(before["speed_kmh"]) < .1 and before["brake"] > .98,
        "ai_off": not before["ai_controlled"],
        "camera_unclaimed": not before["camera_active"],
        "ideal_line_off": not before["ideal_line_shown"],
        "reviewed_csp_source": actual == expected_source,
    }
    return before, window, actual, requirements


def run(args):
    protocol = load_frozen(args.protocol)
    capture = protocol["capture"]
    source = capture["csp_sha256"]
    warning = load_frozen(Path(protocol["warning_end_protocol"]))
    fusion = load_frozen(Path(warning["fusion_protocol"]))
    base = load_frozen(Path(fusion["base_protocol"]))
    track = base["proposed_track"]
    if fusion["development_rehearsal"]:
        raise ValueError("Independent capture requires an independent evaluation protocol")
    camera = Path(fusion["camera_config"])
    args.output.mkdir(parents=True, exist_ok=False)
    recorder = None
    result = dict(status="failed", track=track, actor_control_sent=False, seed=None)
    try:
        before, window, actual, requirements = check_session(track, source)
        result.update(engine_pid=window.pid, before=before, source_sha256=actual,
                      readiness=requirements, protocol_sha256=digest(args.protocol))
        if not all(requirements.values()):
            result.update(status="not_ready", termination="session_preflight")
            return
        if args.preflight:
            result.update(status="ready", termination="session_preflight")
            return

        mount = read(camera)
        recorder = CSPVisionRecorder(window.hwnd, mount, max_frames=capture["max_frames"],
            period=capture["period_s"], max_bytes=capture["max_bytes"], image_format="bmp",
            shared_transport=True, batched_shared_copy=True, async_frame_persistence=True,
            mmf_metadata=True, preserve_producer_time=True, expected_source_sha256=source,
            expected_track=track, source_body_recording=True)
        recorder.wait_ready(timeout=8.)
        write(args.output/"ready.json", dict(request_id=recorder.request_id, engine_pid=window.pid))
        print("CSP capture ready", flush=True)
        deadline = time.monotonic()+capture["deadline_s"]
        while recorder.thread.is_alive() and not (args.output/"stop").exists() and time.monotonic() < deadline:
            recorder.thread.join(.25)
        timed_out = time.monotonic() >= deadline and recorder.thread.is_alive()
        if recorder.thread.is_alive():
            recorder.request_stop()
        closed = recorder.close()
        if recorder.count:
            archive = recorder.archive_to(args.output/"vision", bundle=True)
            write(args.output/"archive-map.json", {track: str((args.output/"vision").resolve())})
            result.update(status="captured", archive=archive)
        else:
            result["status"] = "empty_capture"
        result.update(frames=recorder.count, closed=closed,
                      termination="capture_deadline" if timed_out else "owned_capture_end")
    except Exception as exc:
        result.update(reason=repr(exc), termination="failed")
        raise
    finally:
        if recorder is not None and recorder.thread.is_alive():
            result["cleanup"] = recorder.close()
        write(args.output/"result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
