"""Compare session heartbeat and native AI flag without changing either source."""
import argparse
import json
import time
from pathlib import Path

from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_teacher_physics import timed_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text(encoding='utf8'))
    if sha256(__file__) != plan['program_sha256']:
        raise ValueError('program hash differs')
    run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),actor=False))
    result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,lifecycle_requests_sent=0)
    rows=[]
    try:
        status_path=Path(plan['session_status']);deadline=time.perf_counter()+plan['duration_seconds'];generation=None
        with ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size) as mapping:
            while time.perf_counter()<deadline:
                before,before_mtime,_=read_session_heartbeat(status_path)
                validate_heartbeat(before,time.time()-before_mtime,mutating=False)
                sample,timing=timed_teacher_snapshot(mapping)
                after,after_mtime,_=read_session_heartbeat(status_path)
                validate_heartbeat(after,time.time()-after_mtime,mutating=False)
                if generation is None:generation=before.get('generation')
                if before.get('generation')!=generation or after.get('generation')!=generation:
                    raise ValueError('session generation changed')
                rows.append(dict(native=sample.authority_record(),copy=timing,
                    csp_before={k:before.get(k) for k in ('generation','epoch','script_clock','ai_controlled','paused','live','speed_kmh','camera_active')},
                    csp_after={k:after.get(k) for k in ('generation','epoch','script_clock','ai_controlled','paused','live','speed_kmh','camera_active')},
                    csp_before_mtime=before_mtime,csp_after_mtime=after_mtime))
                time.sleep(plan['period_seconds'])
        if not rows:raise ValueError('no observations')
        write_json(run/'observations.json',rows)
        pairs={}
        for row in rows:
            key=f"{row['csp_before']['ai_controlled']}/{row['native']['builtin_ai_controlled']}/{row['csp_after']['ai_controlled']}"
            pairs[key]=pairs.get(key,0)+1
        result.update(status='completed',samples=len(rows),observed_before_native_after=pairs,
            distinct_packets=len({r['native']['packet_id'] for r in rows}),generation=generation,
            observations_sha256=sha256(run/'observations.json'),
            scope='Read-only host-bracketed heartbeat and native copies. No independent source time, physics tick, actuator, or proof which source represents control authority.')
    except Exception as e:result.update(error=type(e).__name__,reason=str(e))
    write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
