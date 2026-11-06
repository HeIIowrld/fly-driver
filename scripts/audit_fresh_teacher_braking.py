"""Bind fresh teacher braking windows to original native packet receipts."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording",required=True)
    p.add_argument("--role-receipt",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    root=Path(a.recording)
    teacher=root/"teacher"
    event_path=teacher/"events.jsonl"
    role=read(a.role_receipt)
    out=begin_run(a.output,dict(args=vars(a),program_sha256=sha256(__file__),seed=None,
        role_receipt_sha256=sha256(a.role_receipt),vehicle_control_sent=False,fit_allowed=False))
    result=dict(status="failed",fit_allowed=False,student_control=False)
    try:
        proof=read(teacher/"result.json")
        if (proof["status"]!="curriculum_completed" or sha256(event_path)!=proof["events_sha256"]
                or role["fit_allowed"] is not False or role["student_control"] is not False):
            raise ValueError("Completed exact evaluation-only teacher record required")
        samples={}
        requests=[]
        for line in event_path.open(encoding="utf-8"):
            e=json.loads(line)
            if e["kind"]=="raw_pair":
                n=e["native"]
                raw=base64.b64decode(n["prefix_base64"],validate=True)
                if hashlib.sha256(raw).hexdigest()!=n["prefix_sha256"]:
                    raise ValueError("Original native hash differs")
                row=decode_physics_prefix(raw,n["received_monotonic_seconds"])
                if row.packet_id!=n["packet_id"]:
                    raise ValueError("Original native packet id differs")
                samples.setdefault(row.packet_id,(row.received_monotonic_seconds,row.fields))
            if (e["kind"]=="teacher_request" and e["request"]["driver_accepted"]
                    and e["unlimited"]==e["limited"] and not e["automatic_shift_coast"]):
                requests.append(e["request"])
        ordered=sorted(samples.values(),key=lambda v:v[0])
        when=np.array([t for t,_ in ordered])
        speed=np.array([f["speedKmh"] for _,f in ordered])
        brake=np.array([f["brake"] for _,f in ordered])
        starts=np.array([v["driver_call_started_perf_counter_seconds"] for v in requests])
        ends=np.array([v["driver_call_finished_perf_counter_seconds"] for v in requests])
        control=np.array([v["requested_control"]["brake"] for v in requests])
        idx=np.searchsorted(when,starts)-1
        mask=(idx>=0)&(starts-when[idx]<=.05)&(speed[idx]>=100)&(control>.1)
        samples_used=[]
        for j in np.flatnonzero(mask):
            post=(when>=ends[j])&(when<=ends[j]+.10)
            if post.any():
                samples_used.append(dict(request_index=int(j),sdk_end=float(ends[j]),
                    native_before_kmh=float(speed[idx[j]]),request_brake=float(control[j]),
                    max_native_brake_100ms=float(brake[post].max()),
                    min_native_speed_100ms=float(speed[post].min())))
        write_json(out/"sampled-windows.json",samples_used)
        result.update(status="completed_teacher_high_speed_diagnostic",native_packets=len(samples),
            peak_speed_kmh=float(speed.max()),genuine_teacher_requests=len(requests),
            high_speed_brake_requests=int(mask.sum()),requests_with_native_feedback=len(samples_used),
            windows_with_native_brake_above_point1=sum(v["max_native_brake_100ms"]>.1 for v in samples_used),
            events_sha256=proof["events_sha256"],sampled_windows_sha256=sha256(out/"sampled-windows.json"))
    except Exception as exc:
        result["reason"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
