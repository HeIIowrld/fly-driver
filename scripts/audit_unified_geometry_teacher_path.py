"""Post-hoc teacher-path direction versus original graph intent; never actor input."""
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
    for name in ('candidate','teacher-eval','teacher-audit','graph-replay','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    a=p.parse_args()
    candidate,teacher,audit,replay=map(Path,(a.candidate,a.teacher_eval,a.teacher_audit,a.graph_replay))
    c,t,independent,r=[read(x/'result.json') for x in (candidate,teacher,audit,replay)]
    event_hash=sha256(candidate/'recording/events.jsonl')
    tm=read(teacher/'manifest.json')['config'];rm=read(replay/'manifest.json')['config']
    im=read(audit/'manifest.json')['config']
    if (c['role'] != a.role or c['fit_allowed'] is not False or t['role'] != a.role
            or r['role'] != a.role or t['status'] != 'completed_teacher_path_evaluation'
            or independent['status'] != 'completed' or r['status'] != 'completed_unified_flight_offline_replay'
            or tm['events_sha256'] != event_hash or rm['events_sha256'] != event_hash
            or im['events_sha256'] != event_hash or im['evaluation_sha256'] != t['evaluation_sha256']
            or sha256(teacher/'evaluation.npz') != t['evaluation_sha256']
            or sha256(replay/'intents.npz') != r['intents_sha256']
            or sha256(replay/'geometry-readiness.npz') != r['geometry_readiness_sha256']):
        raise ValueError('Exact role-pinned original evaluation and graph replay required')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,events_sha256=event_hash,
        teacher_sha256=t['evaluation_sha256'],teacher_audit_sha256=sha256(audit/'result.json'),
        graph_sha256=r['intents_sha256'],readiness_sha256=r['geometry_readiness_sha256'],
        scope='post-hoc teacher path as reference only; never actor input',optimizer_updates=0))
    with np.load(teacher/'evaluation.npz',allow_pickle=False) as f:
        target=f['target'][:,2,1].copy();eligible=f['informative'].copy();clock=f['decision_clock'].copy()
    with np.load(replay/'intents.npz',allow_pickle=False) as f:
        intents=f['intents'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    with np.load(replay/'geometry-readiness.npz',allow_pickle=False) as f:
        ready=f['ready'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    keep=eligible & ready
    def group(mask):
        n=int(mask.sum())
        if not n:return dict(count=0,zero_port_match=None,trained_port_match=None,changed_sign=0)
        desired=-np.sign(target[mask])  # path screen-right positive, raw flight left positive
        baseline=np.sign(intents[1,mask,0]);trained=np.sign(intents[0,mask,0])
        return dict(count=n,zero_port_match=int((baseline==desired).sum()),
            trained_port_match=int((trained==desired).sum()),changed_sign=int((baseline!=trained).sum()),
            baseline_turn_mean=float(intents[1,mask,0].mean()),
            trained_turn_mean=float(intents[0,mask,0].mean()))
    result=dict(status='completed_posthoc_path_direction_diagnostic',role=a.role,
        ready_informative=group(keep),screen_left_path=group(keep&(target<0)),
        screen_right_path=group(keep&(target>0)),original_informative_total=int(eligible.sum()),
        reference='Actual teacher future 20m chord, not necessarily optimal or direct steering target',
        teacher_pose_or_path_actor_input=False,fit_allowed=False,optimizer_updates=0,
        actor_deployed=False,physical_timing_verified=False,
        caution='Correlated frames; direction match is not driving accuracy or an action label.')
    write_json(run/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
