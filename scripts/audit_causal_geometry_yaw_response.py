"""Compare original native yaw with fixed visual-port sensitivity; diagnostic only."""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','body','reference-audit','replay','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    a=p.parse_args()
    candidate,body,audit,replay=map(Path,(a.candidate,a.body,a.reference_audit,a.replay))
    c,b,r,q=[read(x/'result.json') for x in (candidate,body,audit,replay)]
    event_hash=sha256(candidate/'recording/events.jsonl')
    am=read(audit/'manifest.json')['config'];bm=read(body/'manifest.json')['config']
    if (c['role'] != a.role or c['fit_allowed'] is not False or r['role'] != a.role
            or q['role'] != a.role or r['status'] != 'completed_original_candidate_causal_reference_audit'
            or q['status'] != 'completed_unified_flight_offline_replay'
            or am['events_sha256'] != event_hash or bm['events_sha256'] != event_hash
            or sha256(body/'per-view-body.npz') != b['arrays_sha256']
            or sha256(replay/'intents.npz') != q['intents_sha256']
            or sha256(replay/'geometry-readiness.npz') != q['geometry_readiness_sha256']):
        raise ValueError('Original audited matching source and fixed replay required')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,events_sha256=event_hash,
        reference_rows_sha256=sha256(audit/'rows.json'),body_sha256=b['arrays_sha256'],
        intents_sha256=q['intents_sha256'],readiness_sha256=q['geometry_readiness_sha256'],
        yaw_threshold_rad_s=.02,speed_threshold_m_s=1.,optimizer_updates=0,fit_allowed=False))
    rows=read(audit/'rows.json')
    with np.load(body/'per-view-body.npz',allow_pickle=False) as f:
        values,valid,clock=f['current_values'].copy(),f['current_valid'].copy(),f['decisions'].copy()
    with np.load(replay/'intents.npz',allow_pickle=False) as f:
        intent=f['intents'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    with np.load(replay/'geometry-readiness.npz',allow_pickle=False) as f:
        ready=f['ready'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    if not np.array_equal(ready,[row['ready'] for row in rows]):
        raise ValueError('Original readiness audit mismatch')
    geometry=np.array([row['geometry_yaw'] if row['ready'] else 0. for row in rows])
    yaw=values[:,4];speed=values[:,16]; delta=intent[0,:,0]-intent[1,:,0]
    mask=ready & valid[:,4] & valid[:,16] & (speed>1.) & (np.abs(yaw)>.02)
    def score(x,expected_sign):
        if not mask.any():return dict(n=0,sign_accuracy=None,correlation=None)
        y=yaw[mask];v=x[mask]
        return dict(n=int(mask.sum()),positive_yaw=int((y>0).sum()),negative_yaw=int((y<0).sum()),
            sign_accuracy=float(np.mean(np.sign(v)==expected_sign*np.sign(y))),
            correlation=float(np.corrcoef(v,y)[0,1]) if np.std(v)>0 and np.std(y)>0 else None)
    result=dict(status='completed_recorded_geometry_yaw_diagnostic',role=a.role,
        geometry_same_sign_as_yaw=score(geometry,1),
        intent_delta_opposite_yaw=score(delta,-1),
        ready_count=int(ready.sum()),physical_sensor_alignment_verified=False,
        sampling='Same decision index; graph intent also reflects earlier inputs; no lag optimized',
        caution='Native yaw is body feedback not an independent ground-truth label; '
                'successive frames correlated and steering/control calibration unverified.',
        fit_allowed=False,optimizer_updates=0,vehicle_control_sent=False,actor_deployed=False)
    write_json(run/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
