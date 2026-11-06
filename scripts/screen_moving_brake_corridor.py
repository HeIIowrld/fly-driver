"""Choose a real Monza approach with a mapped straight stopping reserve."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.moving_stop_calibration import sdk_to_native_xyz, signed_reserve
from flybrain_drive.sac_unified_student import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh",required=True)
    parser.add_argument("--failed-stop-audit",required=True)
    parser.add_argument("--output",required=True)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--braking-in-progress",action="store_true",
        help="Screen entries during measured native braking with continued original teacher brake")
    phase.add_argument("--pacing-only",action="store_true",
                        help="Screen entry/stop calibration without a recorded near-term teacher brake")
    parser.add_argument("--minimum-kmh",type=int,default=120)
    parser.add_argument("--maximum-kmh",type=int,default=145)
    args=parser.parse_args()
    if not 100<=args.minimum_kmh<args.maximum_kmh<=160:
        raise ValueError("Explicit staged high-speed bounds required")
    root=Path(args.fresh)
    previous=read(args.failed_stop_audit)
    out=begin_run(args.output,dict(args=vars(args),program_sha256=sha256(__file__),seed=None,
        actor_input=False,student_actuation=False,speed_range_kmh=[args.minimum_kmh,args.maximum_kmh],
        lookahead_braking_seconds=None if args.pacing_only else [.12,1.2],minimum_map_clearance_m=2.7,
        native_approach_brake_band="above_point1" if args.braking_in_progress else "at_most_point1",
        independent_stop_time_reserve_seconds=.75))
    result=dict(status="failed",student_live_admitted=False,actor_input=False)
    try:
        if not previous["stationary_verified"] or previous["moving_stop_calibration_passed"]:
            raise ValueError("Original measured exit/stop evidence required")
        geo=read("configs/ac-teacher-geometry-result.json")["geometry_configuration"]
        if sha256(geo["track_csv"])!=geo["track_sha256"]:
            raise ValueError("Privileged track map differs")
        track=np.genfromtxt(geo["track_csv"],delimiter=",",names=True)
        borders=[np.column_stack([track[f"{side}_border_x"],track[f"{side}_border_y"]])
                 for side in ("left","right")]
        raw=root/"demonstration/teacher/events.jsonl"
        source=read(root/"braking-native-audit-01/result.json")
        if sha256(raw)!=source["events_sha256"]:
            raise ValueError("Original teacher events differ")
        sampled=root/"braking-native-audit-01/sampled-windows.json"
        if sha256(sampled)!=source["sampled_windows_sha256"]:
            raise ValueError("Original brake windows differ")
        labels=[v for v in read(sampled) if v["request_brake"]>.25
                and v["max_native_brake_100ms"]>.1]
        brake_times=np.array([x["sdk_end"] for x in labels])
        poses=[]
        for line in raw.open(encoding="utf-8"):
            e=json.loads(line)
            if e["kind"]!="raw_pair":
                continue
            sdk=e["sdk"]
            payload=base64.b64decode(sdk["wire_base64"],validate=True)
            if hashlib.sha256(payload).hexdigest()!=sdk["wire_sha256"]:
                raise ValueError("Original SDK bytes differ")
            packet=json.loads(payload)["privileged_teacher"]
            if packet["actor_input"] is not False:
                raise ValueError("SDK position requires teacher-only authority")
            fields=packet["raw"]
            poses.append(dict(host_time=e["host_time"],source_sequence=sdk["sequence"],
              world_position_sdk_xyz_m=[fields["world_position_"+k] for k in "xyz"],
              spline_progress=fields["NormalizedSplinePosition"],wire_sha256=sdk["wire_sha256"],
              speed_kmh=e["native"]["fields"]["speedKmh"],brake=e["native"]["fields"]["brake"]))
        candidates=[]
        for i, row in enumerate(poses[2:],start=2):
            speed=row["speed_kmh"]
            if not (args.minimum_kmh<=speed<=args.maximum_kmh
                    and .06<=row["spline_progress"]<=.94):
                continue
            if (row["brake"]>.1) != args.braking_in_progress:
                continue
            t=row["host_time"]
            j=int(np.searchsorted(brake_times,t+.12))
            if not args.pacing_only and (j==len(labels) or brake_times[j]-t>1.2):
                continue
            if not .03 <= t-poses[i-2]["host_time"] <= .12:
                continue
            p=np.asarray(row["world_position_sdk_xyz_m"][:2])
            direction=p-np.asarray(poses[i-2]["world_position_sdk_xyz_m"][:2])
            if np.linalg.norm(direction)<.1:
                continue
            direction/=np.linalg.norm(direction)
            reserve=previous["stop_distance_m"]*(speed/previous["entry_speed_kmh"])**2+speed/3.6*.75+10
            distance=np.arange(0,reserve+1,1.)
            margin=signed_reserve(p+distance[:,None]*direction,borders)
            candidates.append(dict(approach=row,onset=None if args.pacing_only else labels[j],straight_length_m=float(reserve),
                predicted_minimum_center_clearance_m=float(margin.min()),forward_sdk_xy=direction.tolist()))
        write_json(out/"approach-candidates.json",candidates)
        passing=[v for v in candidates if v["predicted_minimum_center_clearance_m"]>=2.7
                 and (args.pacing_only or v["approach"]["speed_kmh"]<=145)]
        result.update(screened=len(candidates),passing=len(passing),
                      best_clearance_m=max((v["predicted_minimum_center_clearance_m"] for v in candidates),default=None))
        if not passing:
            result.update(status="no_eligible_mapped_braking_approach")
        else:
            chosen=passing[0]
            result.update(status="completed_diagnostic_teacher_moving_reference",**chosen,
                native_center_xyz_m=sdk_to_native_xyz(chosen["approach"]["world_position_sdk_xyz_m"]),
                forward_native_xz=[chosen["forward_sdk_xy"][1],chosen["forward_sdk_xy"][0]],
                native_stop_corridor_verified=False,native_graphics_registration_verified=False,
                route_geometry_sha256=geo["track_sha256"],
                teacher_to_student_authority_verified=False,pacing_only=args.pacing_only,
                approach_phase="active_braking" if args.braking_in_progress else "before_brake")
            if not args.pacing_only:
                audit_path=root/"braking-native-audit-01/result.json"
                result.update(track="monza",natural_braking_source=dict(events=str(raw),
                    audit=str(audit_path),audit_sha256=sha256(audit_path),samples=str(sampled)))
    except Exception as error:
        result["reason"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
