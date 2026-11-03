"""Derive native AI offset from pinned CSP source and inspect preserved runs."""
import argparse
import ast
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def reviewed_offset(path):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SPageFilePhysics')
    values={target.id:n.value for n in cls.body if isinstance(n,ast.Assign)
            for target in n.targets if isinstance(target,ast.Name)}
    if ast.literal_eval(values['_pack_'])!=4 or not isinstance(values['_fields_'],ast.List):
        raise ValueError('unexpected ABI declaration')
    offset=0
    for item in values['_fields_'].elts:
        name=ast.literal_eval(item.elts[0]);spec=item.elts[1];count=1
        if isinstance(spec,ast.BinOp) and isinstance(spec.op,ast.Mult):
            count=ast.literal_eval(spec.right);spec=spec.left
        if not isinstance(spec,ast.Name) or spec.id not in ('c_float','c_int32') or type(count) is not int or count<1:
            raise ValueError('unsupported ABI field before target')
        if name=='isAIControlled':return offset
        offset+=4*count
    raise ValueError('AI field absent')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text(encoding='utf8'))
    if sha256(__file__)!=plan['program_sha256']:raise ValueError('program mismatch')
    for path,digest in plan['sources'].items():
        if sha256(path)!=digest:raise ValueError('source mismatch: '+path)
    run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),actor=False))
    result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False)
    try:
        offset=reviewed_offset(plan['csp_reference']);old=json.loads(Path(plan['old_native']).read_text(encoding='utf8'));motion=json.loads(Path(plan['motion_native']).read_text(encoding='utf8'))
        before=json.loads(Path(plan['motion_before']).read_text(encoding='utf8'));after=json.loads(Path(plan['motion_after']).read_text(encoding='utf8'));program=Path(plan['motion_program']).read_text(encoding='utf8')
        if offset!=416 or program.index('sampler.start()')>=program.index('child = subprocess.Popen('):raise ValueError('offset or preserved execution ordering mismatch')
        def counts(rows):
            values=[r['authority']['builtin_ai_controlled'] for r in rows]
            if not values or any(type(v) is not bool for v in values):raise ValueError('invalid flag rows')
            return dict(samples=len(values),true=sum(values),false=len(values)-sum(values),first=values[0],last=values[-1])
        result.update(status='completed',independent_source_ai_offset=offset,
            old_stationary=counts(old),later_motion_window=counts(motion),
            later_session_before={k:before.get(k) for k in ('generation','ai_controlled','expert_probe_active','paused')},
            later_session_after={k:after.get(k) for k in ('generation','ai_controlled','expert_probe_active','paused')},
            recording_starts_before_teacher_process_launch=True,
            scope='Pinned external source parsed as syntax only. Historical runs; observation order does not prove the flag update cause or physical-tick simultaneity.')
    except Exception as exc:result.update(error=type(exc).__name__,reason=str(exc))
    write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
