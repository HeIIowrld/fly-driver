"""Inspect recorded stationary sensors under offline fixed zero task; no actor calibration."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','zero-task','preview-replay','teacher-eval','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    a=p.parse_args();candidate,quiet,normal,teacher=map(Path,(a.candidate,a.zero_task,a.preview_replay,a.teacher_eval))
    c,q,n,t=[read(x/'result.json') for x in (candidate,quiet,normal,teacher)]
    event_path=candidate/'recording/events.jsonl';event_hash=sha256(event_path)
    if (c['role'] != a.role or c['fit_allowed'] is not False or q['role'] != a.role
            or n['role'] != a.role or t['role'] != a.role or q['fixed_zero_command_diagnostic'] is not True
            or q['visual_preview_command_used'] is not False
            or any(read(x/'manifest.json')['config']['events_sha256'] != event_hash for x in (quiet,normal,teacher))
            or sha256(quiet/'intents.npz') != q['intents_sha256']
            or sha256(normal/'intents.npz') != n['intents_sha256']
            or sha256(normal/'geometry-readiness.npz') != n['geometry_readiness_sha256']
            or sha256(teacher/'evaluation.npz') != t['evaluation_sha256']):
        raise ValueError('Original matched role, fixed zero task, and evidence required')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,events_sha256=event_hash,
        zero_task_sha256=q['intents_sha256'],normal_sha256=n['intents_sha256'],
        teacher_sha256=t['evaluation_sha256'],stationary_min_duration_s=1.,
        scope='post-hoc output counterfactual, no calibration or actor',optimizer_updates=0))
    events=[json.loads(line) for line in event_path.read_text(encoding='utf8').splitlines()]
    with np.load(quiet/'intents.npz',allow_pickle=False) as f:
        quiet_intent=f['intents'][0].copy();times=f['decision_clock'].copy()
    with np.load(normal/'intents.npz',allow_pickle=False) as f:
        normal_intent=f['intents'][0].copy();np.testing.assert_array_equal(times,f['decision_clock'])
    with np.load(normal/'geometry-readiness.npz',allow_pickle=False) as f:
        ready=f['ready'].copy();np.testing.assert_array_equal(times,f['decision_clock'])
    with np.load(teacher/'evaluation.npz',allow_pickle=False) as f:
        target=f['target'][:,2,1].copy();eligible=f['informative'].copy();np.testing.assert_array_equal(times,f['decision_clock'])
    stationary=[]
    for i,event in enumerate(events):
        native=event['native'];raw=base64.b64decode(native['prefix_base64'],validate=True)
        sample=decode_physics_prefix(raw,float(native['received_monotonic_seconds']))
        if event['index'] != i or event['decision_monotonic_seconds'] != times[i] or sample.packet_id != native['packet_id']:
            raise ValueError('Original native/time identity changed')
        f=sample.fields;age=times[i]-sample.received_monotonic_seconds
        stationary.append(0 <= age <= .05 and abs(f['speedKmh'])<=.1
            and max(map(abs,f['wheelAngularSpeed']))<=.05 and max(map(abs,f['localAngularVel']))<=.02)
    spans=[];start=None
    for i,good in enumerate(stationary+[False]):
        if good and start is None:start=i
        if not good and start is not None:
            if times[i-1]-times[start]>=1.:spans.append((start,i-1))
            start=None
    if not spans:raise ValueError('No original stationary span lasting at least one second')
    start,end=max(spans,key=lambda pair:times[pair[1]]-times[pair[0]])
    indices=np.arange(start,end+1);selected=indices[times[indices]-times[start]>=1.]
    if not len(selected):raise ValueError('No stationary observations after one second')
    observed=quiet_intent[selected]
    # Output-only recentering is intentionally counterfactual, never installed in graph.
    baseline=float(np.mean(np.arctanh(observed[:,0])))
    if np.any(np.abs(normal_intent[:,0])>=1):raise ValueError('Finite inverse tanh required')
    hypothetical=np.tanh(np.arctanh(normal_intent[:,0])-baseline)
    evaluate=eligible & ready & (np.arange(len(times))>end)
    expected=-np.sign(target[evaluate])
    result=dict(status='completed_stationary_zero_task_diagnostic',role=a.role,
        stationary_start=int(start),stationary_end=int(end),stationary_duration_s=float(times[end]-times[start]),
        stationary_after_one_second_count=int(len(selected)),stationary_raw_mean=observed.mean(0).tolist(),
        stationary_raw_min=observed.min(0).tolist(),stationary_raw_max=observed.max(0).tolist(),
        lateral_preactivation_hypothetical_offset=baseline,later_informative_count=int(evaluate.sum()),
        original_path_direction_matches=int((np.sign(normal_intent[evaluate,0])==expected).sum()),
        hypothetical_direction_matches=int((np.sign(hypothetical[evaluate])==expected).sum()),
        continuous_real_vehicle_validation=False,fit_allowed=False,optimizer_updates=0,
        vehicle_control_sent=False,actor_deployed=False,
        warning='Zero task imposed counterfactually over original teacher-driven motion; '
                'same-state stationary offset is not validated calibration or a steering target.')
    write_json(run/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
