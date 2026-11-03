"""Replay causal stationary reference on original candidate sensors; no fitting/control."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_geometry_feedback import CausalGeometryVisualPort
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


class ZeroBase(nn.Module):
    def forward(self,traces,body,valid):
        return torch.zeros(1,2)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','raw-audit','visual','visual-audit','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    a=p.parse_args()
    candidate,raw,visual,audit=map(Path,(a.candidate,a.raw_audit,a.visual,a.visual_audit))
    outer,rr,vr,ar=[read(p/'result.json') for p in (candidate,raw,visual,audit)]
    events_path=candidate/'recording/events.jsonl'; event_hash=sha256(events_path)
    vm=read(visual/'manifest.json')['config'];rm=read(raw/'manifest.json')['config']
    if (outer['role'] != a.role or outer['fit_allowed'] is not False or rr['status'] != 'completed'
            or vr['status'] != 'completed' or ar['status'] != 'completed'
            or ar['full_chronological_comparison']['observations_compared'] != outer['decisions']
            or rm['events_sha256'] != event_hash or vm['events_sha256'] != event_hash
            or sha256(visual/'responses.npz') != vr['responses_sha256']):
        raise ValueError('Exact audited role-pinned original candidate required')
    camera=Path('configs/ac-csp-exterior-front-v3-retina.json')
    registration=visual/'retina-registration.npz'
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,events_sha256=event_hash,
        responses_sha256=vr['responses_sha256'],registration_sha256=sha256(registration),
        camera_sha256=sha256(camera),optimizer_updates=0,fit_allowed=False))
    with np.load(registration,allow_pickle=False) as f:
        pool=VisualSelfMotionPool.from_registration(read(camera),{k:f[k].copy() for k in f.files})
    with np.load(visual/'responses.npz',allow_pickle=False) as f:
        motion,times,captures=f['motion_activity'].copy(),f['decision_monotonic_seconds'].copy(),f['capture_ids'].copy()
    events=[json.loads(row) for row in events_path.read_text(encoding='utf8').splitlines()]
    if len(events) != outer['decisions'] or motion.shape != (len(events),4,8,721):
        raise ValueError('Original complete four-view history required')
    port=CausalGeometryVisualPort(ZeroBase(),pool,2)
    rows=[]
    for i,event in enumerate(events):
        native=event['native'];now=float(times[i]);capture=int(captures[i])
        if event['index'] != i or event['camera']['capture_id'] != capture or event['decision_monotonic_seconds'] != now:
            raise ValueError('Original capture identity or time changed')
        data=base64.b64decode(native['prefix_base64'],validate=True)
        sample=decode_physics_prefix(data,float(native['received_monotonic_seconds']))
        if sample.packet_id != native['packet_id']:
            raise ValueError('Original native packet changed')
        active=port.observe(motion[i],np.ones(4,dtype=bool),sample,capture_id=capture,decision_time=now, camera_receipt_time=float(event['camera']['received']))
        rows.append(dict(index=i,capture_id=capture,packet_id=sample.packet_id,
            ready=active,reason=port.reason,reference_available=port.reference.value is not None,
            geometry_yaw=float(port.current[0,2]) if active else None))
    write_json(run/'rows.json',rows)
    ready=[row for row in rows if row['ready']]
    result=dict(status='completed_original_candidate_causal_reference_audit',role=a.role,
        observations=len(rows),ready_count=len(ready),first_ready_index=ready[0]['index'] if ready else None,
        reason_counts={key:sum(row['reason']==key for row in rows) for key in sorted({row['reason'] for row in rows})},
        actor_deployed=False,fit_allowed=False,optimizer_updates=0,vehicle_control_sent=False,
        camera_exposure_or_body_physical_time_verified=False,
        scope='Original recorded host receipts; not proof of real-time or closed-loop operation')
    write_json(run/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
