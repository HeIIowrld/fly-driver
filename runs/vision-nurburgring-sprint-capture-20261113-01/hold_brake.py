"""Hold the reviewed virtual brake briefly while the Nürburgring Sprint session is stationary."""
import argparse
import configparser
import json
import os
from pathlib import Path
import time

from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_native_physics import NativePhysicsReader, repeated_snapshot
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.graph import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    axis=json.loads(Path("configs/ac-axis-probe.json").read_bytes())
    profile=Path(axis["profile_path"])
    pin=sha256(profile)
    controls=configparser.ConfigParser(interpolation=None,strict=False)
    controls.read(profile,encoding="utf-8-sig")
    requirements={("HEADER","INPUT_METHOD"):"WHEEL",("CONTROLLERS","PGUID0"):axis["instance_guid"],
                  ("STEER","JOY"):"0",("STEER","AXLE"):"0",
                  ("THROTTLE","JOY"):"0",("THROTTLE","AXLE"):"1",
                  ("THROTTLE","MIN"):"-1",("THROTTLE","MAX"):"1",
                  ("BRAKES","JOY"):"0",("BRAKES","AXLE"):"2",
                  ("BRAKES","MIN"):"-1",("BRAKES","MAX"):"1"}
    if any(controls.get(*k,fallback=None)!=v for k,v in requirements.items()):
        raise ValueError("Active vJoy axis mapping differs")
    devices=inspect_vjoy_controllers()
    if devices["candidate_ac_joy_index"]!=0 or [d["instance_guid"] for d in devices["vjoy_devices"]]!=[axis["instance_guid"]]:
        raise ValueError("The active vJoy instance differs")
    sdk=load_vjoy_sdk(axis["dll_path"],axis["dll_sha256"])
    receipt=dict(status="failed",track="ks_nurburgring",control_profile_sha256=pin,
                 source_sha256=sha256(__file__),seed=None)
    session=VJoySession(sdk,axis["device_id"],[AxisBinding(**b) for b in axis["bindings"]])
    status=Path(os.environ["LOCALAPPDATA"])/"flybrain/csp-session/status.json"
    generation=None
    published=False
    try:
        with NativePhysicsReader(timeout=.3) as reader:
            ai=NativeTeacherStatusReader()
            try:
                deadline=time.monotonic()+90.
                while time.monotonic()<deadline and not (out/"release").exists():
                    state,mtime=read_published_json(status)
                    if generation is None:
                        generation=state["generation"]
                    if (not 0 <= time.time()-mtime <= 2 or state["generation"]!=generation
                            or state["track_id"]!="ks_nurburgring" or state["car_id"]!="bmw_z4_gt3"
                            or state["is_online"] or state["ai_controlled"] or state["in_menu"]
                            or abs(state["speed_kmh"])>=.1 or state["gas"]>=.02 or sha256(profile)!=pin):
                        raise ValueError("Stationary Sprint session identity changed")
                    f=repeated_snapshot(reader.mapping).fields
                    if abs(f["speedKmh"])>=.1 or f["gas"]>=.02 or ai.snapshot()["builtin_ai_controlled"]:
                        raise ValueError("Native stationary control condition changed")
                    if not session.owned:
                        session.open()
                    else:
                        session.submit(0.,0.,1.)
                    if f["brake"]>.98 and state["brake"]>.98 and not published:
                        (out/"ready.json").write_text(json.dumps(dict(paused=state["paused"],brake=f["brake"])),encoding="utf-8")
                        print("Brake hold witnessed",flush=True)
                        published=True
                    time.sleep(.045)
            finally:
                ai.close()
        receipt.update(status="completed",brake_witnessed=published,termination="bounded_hold_end")
    except Exception as exc:
        receipt.update(reason=repr(exc),termination="failed")
        raise
    finally:
        if session.owned:
            session.close()
        receipt.update(actuator_messages_sent=session.submit_attempts,device_free=sdk.GetVJDStatus(axis["device_id"])==1)
        (out/"result.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
