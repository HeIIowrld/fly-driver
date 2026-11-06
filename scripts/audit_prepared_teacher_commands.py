"""Independently recompute past-command binding and discrepancy statistics."""
import argparse,json,math
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.comparison);r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];args=cfg['arguments'];source=Path(args['shadow']);teacher=source/'teacher';replay=Path(args['preroll_replay'])
    assert r['status']=='completed_prepared_teacher_command_comparison' and r['training_admissible'] is False
    for file,key in [('pairings.json','pairings_sha256'),('commands.json','commands_sha256'),('comparison.npz','arrays_sha256')]:assert sha256(root/file)==r[key]
    assert sha256(root/'executed-entrypoint.py')==cfg['program_sha256'] and sha256(teacher/'events.jsonl')==cfg['teacher_events_sha256']
    assert sha256(replay/'result.json')==cfg['preroll_result_sha256'] and sha256(source/'result.json')==cfg['shadow_result_sha256']
    raw=[json.loads(line) for line in (teacher/'events.jsonl').read_text(encoding='utf-8').splitlines()];commands=[e for e in raw if e['kind']=='teacher_request'];prepared=read(replay/'original-preroll.json');pairs=read(root/'pairings.json')
    with np.load(root/'comparison.npz') as z:
        times=z['decision_time'];prediction=z['prediction'];targets=z['requested_target'];mask=z['comparison_eligible'];indices=z['teacher_command_index']
        expected=[]
        for i,t in enumerate(times):
            past=[j for j,c in enumerate(commands) if c['request']['driver_call_finished_perf_counter_seconds']<=t]
            j=past[-1] if past else -1;valid=j>=0 and t-commands[j]['request']['driver_call_finished_perf_counter_seconds']<=.06
            assert pairs[i]['eligible']==valid and bool(mask[i])==(valid and prepared[i]['prepared'])
            if valid:
                c=commands[j];v=c['request']['requested_control'];target=[v['steering']/cfg['steering_request_scale'],v['throttle'],v['brake']]
                assert c['request']['driver_accepted'] is True and np.array_equal(targets[i],np.asarray(target,dtype=targets.dtype)) and indices[i]==j
            else:assert indices[i]==-1 and np.array_equal(targets[i],np.zeros(3))
            expected.append(bool(mask[i]))
        checks=[('all_prepared',mask,r['all_prepared'])]
        for stage,stats in r['stages'].items():checks.append((stage,mask&np.array([j>=0 and commands[j]['stage']==int(stage) for j in indices]),stats))
        for name,selected,stats in checks:
            ids=np.flatnonzero(selected);groups={}
            for i in ids:groups.setdefault(int(indices[i]),[]).append(int(i))
            assert stats['events']==len(ids) and stats['unique_teacher_commands']==len(groups)
            for k in range(3):
                error=[float(prediction[i,k])-float(targets[i,k]) for i in ids]
                rmse=math.sqrt(math.fsum(e*e for e in error)/len(error));mae=math.fsum(abs(e) for e in error)/len(error)
                balanced=math.sqrt(math.fsum(math.fsum((float(prediction[i,k])-float(targets[i,k]))**2 for i in g)/len(g) for g in groups.values())/len(groups))
                assert abs(rmse-stats['rmse'][k])<1e-12 and abs(mae-stats['mae'][k])<1e-12 and abs(balanced-stats['command_balanced_rmse'][k])<1e-12
            material=[i for i in ids if abs(targets[i,0])>=.05 and abs(prediction[i,0])>=.05]
            assert len(material)==stats['material_steering_events'] and sum(prediction[i,0]*targets[i,0]<0 for i in material)==stats['opposite_steering_events']
            assert sum(prediction[i,1]>.1 and prediction[i,2]>.1 for i in ids)==stats['simultaneous_pedals_above_point1']
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),comparison_result_sha256=sha256(root/'result.json'),seed=None,optimizer_updates=0,vehicle_control_sent=False))
    write_json(out/'result.json',dict(status='completed_independent_prepared_command_comparison_audit',events=len(times),compared_events=int(mask.sum()),unique_commands=len(set(indices[mask])),no_future_commands_verified=True,canonical_steering_scale=cfg['steering_request_scale'],training_admissible=False,actor_qualified=False,vehicle_control_sent=False,termination_reason='Independent raw-command search, normalized targets, preparation/missing masks and all stagewise discrepancy metrics verified; requested-command discrepancy only, not same-state optimal action or physical application truth.'))

if __name__=='__main__':main()
