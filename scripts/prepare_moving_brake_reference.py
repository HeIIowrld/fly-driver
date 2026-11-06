"""Choose a prospectively defined moving brake reference from original teacher bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fresh",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--minimum-kmh",type=int,default=144)
    p.add_argument("--maximum-kmh",type=int,default=300)
    p.add_argument("--require-unbraked-approach",action="store_true")
    a=p.parse_args()
    if not 100<=a.minimum_kmh<a.maximum_kmh<=300:
        raise ValueError("Explicit diagnostic high-speed range required")
    root=Path(a.fresh)
    qa=read(root/"qualification-02/result.json")
    raw=read(root/"braking-native-audit-01/result.json")
    out=begin_run(a.output,dict(args=vars(a),program_sha256=sha256(__file__),seed=None,
      vehicle_control_sent=False,actor_input=False,
      entry_rule="first 3 consecutive >.25 brake requests in declared speed range; lead .25s"))
    result=dict(status="failed",actor_input=False,student_live_admitted=False)
    try:
        if qa["status"]!="qualified_bounded_camera_body_alignment":
            raise ValueError("Qualified original reference episode required")
        sampled=root/"braking-native-audit-01/sampled-windows.json"
        if sha256(sampled)!=raw["sampled_windows_sha256"]:
            raise ValueError("Original verified braking windows differ")
        rows=read(sampled)
        targets=[]
        for i in range(len(rows)-2):
            cluster=rows[i:i+3]
            if all(x["request_brake"]>.25 and a.minimum_kmh<=x["native_before_kmh"]<=a.maximum_kmh for x in cluster) and all(
                    0 < b["sdk_end"]-x["sdk_end"] <= .05 for x,b in zip(cluster,cluster[1:])):
                targets.append(cluster[0])
        if not targets:raise ValueError("No real fast contiguous braking window")
        events=root/"demonstration/teacher/events.jsonl"
        if sha256(events)!=raw["events_sha256"]:
            raise ValueError("Original teacher record differs")
        poses=[]
        for line in events.open(encoding="utf-8"):
            e=json.loads(line)
            if e["kind"]!="raw_pair":
                continue
            sdk=e["sdk"]
            wire=base64.b64decode(sdk["wire_base64"],validate=True)
            if hashlib.sha256(wire).hexdigest()!=sdk["wire_sha256"]:
                raise ValueError("Original privileged wire hash differs")
            v=json.loads(wire)["privileged_teacher"]
            if v["actor_input"] is not False:raise ValueError("Privileged path cannot reach actor")
            w=v["raw"]
            poses.append(dict(host_time=e["host_time"],source_sequence=sdk["sequence"],
                 speed_kmh=e["native"]["fields"]["speedKmh"],brake=e["native"]["fields"]["brake"],
                 world_position_sdk_xyz_m=[w["world_position_"+k] for k in "xyz"],
                 spline_progress=w["NormalizedSplinePosition"],wire_sha256=sdk["wire_sha256"]))
        times=np.array([v["host_time"] for v in poses])
        target=None
        for v in targets:
            onset=v["sdk_end"]
            j=int(np.searchsorted(times,onset-.25)-1)
            if j<2 or onset-.25-times[j]>.08:continue
            if a.require_unbraked_approach and not (
                    poses[j]["brake"]<=.1 and a.minimum_kmh<=poses[j]["speed_kmh"]<=a.maximum_kmh):
                continue
            target=v
            break
        if target is None:
            raise ValueError("No approach satisfying the declared speed/brake bounds")
        coords=np.array([v["world_position_sdk_xyz_m"] for v in poses])
        direction=coords[j,[0,2]]-coords[j-2,[0,2]]
        distance=np.linalg.norm(direction)
        if not np.isfinite(distance) or distance<.05:
            raise ValueError("Original approach direction is unresolved")
        direction/=distance
        write_json(out/"original-reference-poses.json",[
            p for p in poses if onset-.6<=p["host_time"]<=onset+.75])
        result.update(status="completed_diagnostic_teacher_moving_reference",
            onset=target,approach=poses[j],forward_sdk_xz=direction.tolist(),
            approach_to_onset_seconds=onset-times[j],
            native_graphics_registration_verified=False,teacher_to_student_authority_verified=False,
            native_stop_corridor_verified=False,unseen_geometry=False,
            reference_sha256=sha256(out/"original-reference-poses.json"))
    except Exception as exc:
        result["reason"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
