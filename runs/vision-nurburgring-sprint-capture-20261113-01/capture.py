"""Record a frozen independent Nürburgring Sprint evaluation in an offline cockpit."""
import argparse
import configparser
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0, str(RUNS / "vision-corner-left-risk-reallocation-20261113-01"))
from capture_highlands import PROTOCOL
from replay_independent import load_fixed_policy, validate_transfer
from fixed_left import old, digests
from scripts.capture_independent_corner import check_session
from flybrain_drive.ac_csp_vision import CSPVisionRecorder

TRACK = "ks_nurburgring"
LAYOUT = "layout_sprint_a"
POLICY = RUNS / "vision-corner-phase-veto-frontier-20261113-01/frozen.json"
LINE = RUNS / "vision-optimized-line-teacher-20261113-01/readout-frozen.json"


def verify_race(path):
    data = path.read_bytes()
    cfg = configparser.ConfigParser(interpolation=None,strict=False)
    cfg.read_string(data.decode("utf-8-sig"))
    expected = {("RACE","TRACK"):TRACK,("RACE","CONFIG_TRACK"):LAYOUT,
                ("RACE","MODEL"):"bmw_z4_gt3",("REMOTE","ACTIVE"):"0",("REPLAY","ACTIVE"):"0"}
    if any(cfg.get(*key,fallback=None) != value for key,value in expected.items()):
        raise ValueError("Offline Sprint configuration does not match")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-ini",type=Path,required=True)
    args = parser.parse_args()
    reference, *_ = load_fixed_policy()
    spline = validate_transfer(TRACK,LAYOUT)
    for path in (POLICY,LINE):
        receipt = old.read(path)
        for source, checksum in receipt["sources"].items():
            if old.digest(Path(source)) != checksum:
                raise ValueError("A frozen input changed")
        used = receipt.get("selection_recordings",receipt.get("comparisons",[]))
        if TRACK in used or TRACK in receipt.get("development",[]):
            raise ValueError("Sprint is already used by the selected detector")
    prior_config = verify_race(args.race_ini)
    (OUT/"race-before.ini").write_bytes(prior_config)
    c = old.read(PROTOCOL)
    camera = Path(old.read(Path(old.read(Path(c["warning_end_protocol"]))["fusion_protocol"]))["camera_config"])
    c = c["capture"]
    state,window,source,ready = check_session(TRACK,c["csp_sha256"])
    result = dict(status="failed",track=TRACK,layout=LAYOUT,actor_control_sent=False,
                  requirements=ready,engine_pid=window.pid,generation=state["generation"],
                  source_revision=old.revision(),seed=None,
                  inputs=digests((POLICY,LINE,reference,spline,PROTOCOL,camera,Path(__file__))))
    recorder = None
    try:
        if not all(ready.values()):
            raise ValueError(f"Session did not pass capture preflight: {ready}")
        recorder = CSPVisionRecorder(window.hwnd,old.read(camera),max_frames=c["max_frames"],
            period=c["period_s"],max_bytes=c["max_bytes"],image_format="bmp",shared_transport=True,
            batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,
            expected_source_sha256=source,expected_track=TRACK,source_body_recording=True)
        start = time.monotonic()
        while not recorder.count and recorder.thread.is_alive() and time.monotonic()-start < 18.:
            recorder.thread.join(.1)
        if not recorder.count:
            raise ValueError(f"Camera stream produced no frames: {recorder.error}")
        old.write(OUT/"ready.json",dict(request_id=recorder.request_id,first_frames=recorder.count,
                  producer=str(recorder.directory),consumer=str(recorder.local)))
        print("Camera ready",recorder.request_id,flush=True)
        while recorder.thread.is_alive() and time.monotonic()-start < c["deadline_s"]:
            recorder.thread.join(.25)
        stop = recorder.close()
        result.update(closed=stop,frames=recorder.count,termination="owned_capture_end")
        if recorder.count!=c["max_frames"] or stop["error"] is not None or stop["producer_terminal"]["status"]!="completed":
            raise ValueError("Camera recording did not reach its original completion")
        old.write(OUT/"capture-terminal.json",stop)
        if verify_race(args.race_ini)!=prior_config:
            raise ValueError("The original race configuration changed during recording")
        archive = OUT/"vision"
        result["archive"] = recorder.archive_to(archive,bundle=True)
        mapping = OUT/"archive-map.json"
        old.write(mapping,{TRACK:str(archive)})
        result.update(status="captured",archives_sha256=old.digest(mapping),
                      race_ini_sha256=old.digest(args.race_ini),layout_source="unchanged_active_race_ini")
    except Exception as error:
        result.update(reason=repr(error),termination="failed")
        raise
    finally:
        try:
            if recorder is not None and recorder.thread.is_alive():
                result["cleanup"]=recorder.close()
        finally:
            old.write(OUT/"result.json",result)


if __name__ == "__main__":
    main()
