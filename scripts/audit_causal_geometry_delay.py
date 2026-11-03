"""Compare fixed graph output with recent original geometry/body history; no fitting."""
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
    if (c['role'] != a.role or c['fit_allowed'] is not False or r['role'] != a.role
            or q['role'] != a.role or r['status'] != 'completed_original_candidate_causal_reference_audit'
            or q['status'] != 'completed_unified_flight_offline_replay'
            or read(audit/'manifest.json')['config']['events_sha256'] != event_hash
            or read(body/'manifest.json')['config']['events_sha256'] != event_hash
            or sha256(body/'per-view-body.npz') != b['arrays_sha256']
            or sha256(replay/'intents.npz') != q['intents_sha256']
            or sha256(replay/'geometry-readiness.npz') != q['geometry_readiness_sha256']):
        raise ValueError('Original audited matching fixed replay required')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,events_sha256=event_hash,
        reference_rows_sha256=sha256(audit/'rows.json'),body_sha256=b['arrays_sha256'],
        intents_sha256=q['intents_sha256'],readiness_sha256=q['geometry_readiness_sha256'],
        lags_decisions=[0,1,2,3,4,5],same_rows_for_all_lags=True,
        mask='current speed>1m/s, current |native yaw|>.02rad/s, preceding 5 decisions ready',
        optimizer_updates=0,fit_allowed=False))
    rows=read(audit/'rows.json')
    with np.load(body/'per-view-body.npz',allow_pickle=False) as f:
        values,valid,clock=f['current_values'].copy(),f['current_valid'].copy(),f['decisions'].copy()
    with np.load(replay/'intents.npz',allow_pickle=False) as f:
        intent=f['intents'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    with np.load(replay/'geometry-readiness.npz',allow_pickle=False) as f:
        ready=f['ready'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    if not np.array_equal(ready,[r['ready'] for r in rows]):
        raise ValueError('Exact original readiness mask required')
    geometry=np.array([r['geometry_yaw'] if r['ready'] else 0. for r in rows])
    yaw=values[:,4]; delta=intent[0,:,0]-intent[1,:,0]
    mask=ready & valid[:,4] & valid[:,16] & (values[:,16]>1.) & (np.abs(yaw)>.02)
    mask[:5]=False
    for lag in range(1,6):
        mask[lag:] &= ready[:-lag] & valid[:-lag,4]
    indices=np.flatnonzero(mask)
    def stats(x,y,sign):
        if not len(x):return dict(n=0,sign_agreement=None,correlation=None)
        return dict(n=len(x),sign_agreement=float(np.mean(np.sign(x)==sign*np.sign(y))),
            correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None)
    report=[]
    for lag in range(6):
        old=indices-lag
        report.append(dict(lag_decisions=lag,
            elapsed_seconds_median=float(np.median(clock[indices]-clock[old])) if len(indices) else None,
            intent_vs_past_geometry=stats(delta[indices],geometry[old],-1),
            intent_vs_past_body_yaw=stats(delta[indices],yaw[old],-1)))
    write_json(run/'result.json',dict(status='completed_fixed_lag_diagnostic',role=a.role,
        same_common_rows=int(len(indices)),lags=report,fit_allowed=False,optimizer_updates=0,
        vehicle_control_sent=False,actor_deployed=False,
        warning='Exploratory fixed lags; correlated frames, host time and uncertain physical '
                'alignment. Lag comparisons are not a response latency measurement.'))
    print(json.dumps(report))


if __name__=='__main__':main()
