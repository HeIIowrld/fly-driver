"""Capture an offline Highlands Short run from a verified stationary neutral cockpit."""
import argparse
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"vision-corner-left-risk-reallocation-20261113-01"))
from capture_highlands import LAYOUT, PROTOCOL, TRACK, verify_layout, load_fixed_policy, validate_transfer
from fixed_left import old, digests
from scripts.capture_independent_corner import check_session
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_native_physics import NativePhysicsReader, PHYSICS_PREFIX, PHYSICS_LAYOUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-ini",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    before=verify_layout(args.race_ini)
    policy=load_fixed_policy()[0]
    spline=validate_transfer(TRACK,LAYOUT)
    acquire=old.read(PROTOCOL)
    config=acquire["capture"]
    warning=old.read(Path(acquire["warning_end_protocol"]))
    camera=Path(old.read(Path(warning["fusion_protocol"]))["camera_config"])
    state,window,source,requirements=check_session(TRACK,config["csp_sha256"])
    requirements.pop("stationary_braked")
    with NativePhysicsReader(timeout=.4) as reader:
        sample=reader.receive()
    physics=sample.fields
    decoded=iter(PHYSICS_PREFIX.unpack(sample.prefix_bytes))
    raw={name:tuple(next(decoded) for _ in range(n)) for name,_,n in PHYSICS_LAYOUT}
    requirements.update(not_in_menu=state.get("in_menu") is False,
                        stationary=abs(physics["speedKmh"])<.1 and abs(state["speed_kmh"])<.1,
                        zero_throttle=physics["gas"]<.02 and state["gas"]<.02,
                        neutral=raw["gear"][0]==1)
    result=dict(status="failed",track=TRACK,layout=LAYOUT,actor_control_sent=False,
                engine_pid=window.pid,source_sha256=source,readiness=requirements,
                program_inputs=digests((Path(__file__),policy,spline,PROTOCOL,camera)),
                session_generation=state["generation"],seed=None)
    recorder=None
    try:
        if not all(requirements.values()):
            result.update(status="not_ready",termination="session_preflight")
            return
        recorder=CSPVisionRecorder(window.hwnd,old.read(camera),max_frames=config["max_frames"],
            period=config["period_s"],max_bytes=config["max_bytes"],image_format="bmp",shared_transport=True,
            batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,
            expected_source_sha256=source,expected_track=TRACK,source_body_recording=True)
        recorder.wait_ready(timeout=8.)
        old.write(out/"ready.json",dict(request_id=recorder.request_id,engine_pid=window.pid))
        print("CSP capture ready",flush=True)
        deadline=time.monotonic()+config["deadline_s"]
        while recorder.thread.is_alive() and time.monotonic()<deadline and not (out/"stop").exists():
            recorder.thread.join(.25)
        if recorder.thread.is_alive():
            recorder.request_stop()
        closed=recorder.close()
        result.update(closed=closed,frames=recorder.count,termination="owned_capture_end")
        if recorder.count:
            if verify_layout(args.race_ini)!=before:
                raise ValueError("Active race configuration changed during capture")
            result["archive"]=recorder.archive_to(out/"vision",bundle=True)
            mapping=out/"archive-map.json"
            old.write(mapping,{TRACK:str(out/"vision")})
            result.update(status="captured",race_ini_sha256=old.digest(args.race_ini),
                          archives_sha256=old.digest(mapping),layout_source="unchanged_active_race_ini")
        else:
            result["status"]="empty_capture"
    except Exception as exc:
        result.update(reason=repr(exc),termination="failed")
        raise
    finally:
        if recorder is not None and recorder.thread.is_alive():
            result["cleanup"]=recorder.close()
        old.write(out/"result.json",result)


if __name__=="__main__":
    main()
