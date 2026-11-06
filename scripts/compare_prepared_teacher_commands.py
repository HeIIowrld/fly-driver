"""Retain explicit missing labels and compare prepared student to past teacher commands."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.moving_teacher_command_comparison import accepted_teacher_commands,pair_past_commands,discrepancy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--preroll-replay',required=True);p.add_argument('--preroll-audit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    shadow=Path(a.shadow);replay=Path(a.preroll_replay);audit=Path(a.preroll_audit)
    original=read(shadow/'result.json');rr=read(replay/'result.json');rm=read(replay/'manifest.json')['config'];ar=read(audit/'result.json');am=read(audit/'manifest.json')['config']
    if rr['status']!='completed_moving_native_sensory_interventions' or ar['status']!='completed_independent_moving_sensory_replay_audit' or am['replay_result_sha256']!=sha256(replay/'result.json') or rm['shadow_result_sha256']!=sha256(shadow/'result.json') or sha256(replay/'arrays.npz')!=rr['arrays_sha256']:raise ValueError('Audited same-source recurrent replay required')
    teacher=shadow/'teacher';tr=read(teacher/'result.json');tm=read(teacher/'manifest.json')['config'];events=[json.loads(line) for line in (teacher/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    if sha256(teacher/'events.jsonl')!=tr['events_sha256'] or sha256(teacher/'result.json')!=original['teacher_result_sha256']:raise ValueError('Teacher source changed')
    scale=tm['arguments']['steering_request_scale'];commands=accepted_teacher_commands(events,scale)
    records=read(replay/'original-preroll.json')
    with np.load(replay/'arrays.npz') as z:times=z['decision_time'].copy();pred=z['original_prediction'].copy()
    if len(records)!=len(times) or any(e['decision_time']!=t for e,t in zip(records,times)):raise ValueError('Preparation clock differs')
    pairs=pair_past_commands(times,commands,maximum_age_seconds=.06)
    labels=np.zeros_like(pred);eligible=np.zeros(len(times),dtype=bool);indices=np.full(len(times),-1,dtype=int)
    for i,pair in enumerate(pairs):
        pair.update(event=i,decision_time=float(times[i]),prepared=records[i]['prepared'])
        if pair['eligible']:
            indices[i]=pair['command_index'];labels[i]=commands[indices[i]]['target'];eligible[i]=records[i]['prepared']
        pair['comparison_eligible']=bool(eligible[i])
    if not eligible.any():raise ValueError('No prepared past-command comparisons')
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),helper_sha256=sha256('src/flybrain_drive/moving_teacher_command_comparison.py'),shadow_result_sha256=sha256(shadow/'result.json'),preroll_result_sha256=sha256(replay/'result.json'),preroll_audit_sha256=sha256(audit/'result.json'),teacher_events_sha256=tr['events_sha256'],seed=None,optimizer_updates=0,vehicle_control_sent=False,maximum_command_age_seconds=.06,steering_request_scale=scale,targets='Past driver-accepted requested steering/scale, throttle, brake; not measured game-applied controls or optimal action truth',training_admissible=False))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');write_json(run/'pairings.json',pairs);write_json(run/'commands.json',commands)
    np.savez_compressed(run/'comparison.npz',decision_time=times,prediction=pred,requested_target=labels,comparison_eligible=eligible,teacher_command_index=indices)
    stages={}
    for stage in sorted(set(c['stage'] for c in commands)):
        mask=eligible&np.asarray([indices[i]>=0 and commands[indices[i]]['stage']==stage for i in range(len(times))])
        if mask.any():stages[str(stage)]=discrepancy(pred[mask],labels[mask],indices[mask])
    result=dict(status='completed_prepared_teacher_command_comparison',all_prepared=discrepancy(pred[eligible],labels[eligible],indices[eligible]),stages=stages,events=len(times),prepared_events=sum(e['prepared'] for e in records),unpaired_events=sum(not e['eligible'] for e in pairs),game_applied_control_verified=all(c['game_applied_verified'] for c in commands),training_admissible=False,actor_qualified=False,optimizer_updates=0,vehicle_control_sent=False,pairings_sha256=sha256(run/'pairings.json'),commands_sha256=sha256(run/'commands.json'),arrays_sha256=sha256(run/'comparison.npz'),termination_reason='Prepared original student outputs compared with latest past accepted normal teacher command within60ms; missing/startup/terminal labels retained. Commands are not same-state optimal action or physical application truth; diagnostic only.')
    write_json(run/'result.json',result);print(json.dumps(result))

if __name__=='__main__':main()
