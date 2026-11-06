"""Full fixed replay and same-state/body visual interventions after visual fit."""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from scripts.native_visual_experiment import load_experiment,tensor_state_equal
from scripts.sac_steering_experiment import snapshot,restore
from scripts.train_qualified_sac_unified import context
from scripts.evaluate_precorner_visual_transfer import floor_indices


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);root=Path(a.fit);r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];plan=cfg['plan']
    audit=Path(a.audit)
    assert read(audit/'result.json')['status']=='completed_independent_native_visual_anticipation_arithmetic_audit'
    assert read(audit/'manifest.json')['config']['fit_result_sha256']==sha256(root/'result.json')
    assert r['status']=='completed_native_visual_anticipation_fit'
    assert sha256(root/'visual-anticipation.pt')==r['checkpoint_sha256']
    assert sha256(root/'predictions.npz')==r['predictions_sha256']
    cp=torch.load(root/'visual-anticipation.pt',map_location='cpu',weights_only=True)
    parts=load_experiment(plan['reference_plan']);model,head,dn,optical,port,_=parts[0]
    entries,items,extras,native,valid,pre=parts[2:8]
    for key,module in [('readout',head),('adapter',dn),('body_port',port)]:assert tensor_state_equal(cp[key],module.state_dict())
    old_weight=optical.projection.weight.detach().clone();optical.load_state_dict(cp['optical'])
    new_weight=optical.projection.weight.detach().clone()
    for module in (model,head,dn,optical,port):module.eval().requires_grad_(False)
    prior=Path('runs/native-visual-precorners-20261104/replay-01');prior_result=read(prior/'result.json')
    assert prior_result['status']=='completed_frozen_precorner_visual_transfer'
    for e in range(3):assert sha256(prior/f'episode-{e}.npz')==prior_result['comparisons'][e]['predictions_sha256']
    run=begin_run(a.output,dict(fit=str(root),fit_result_sha256=sha256(root/'result.json'),fit_audit_sha256=sha256(audit/'result.json'),
        program_sha256=sha256(__file__),prior_visual_result_sha256=sha256(prior/'result.json'),entries=entries,seed=None,
        optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,
        scope='Consumed Monza feature-boundary interventions, not pixels/physical counterfactual action truth or unseen corners'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    holder=dict(e=0,i=0,extra=None,neutral=False);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
    def descending(module,args,value):
        steer,power=value;return steer+dn(holder['extra']),power
    def spatial(module,args,value):return value if holder['neutral'] else value+optical(holder['extra'][:,64:])
    def body(module,args,value):
        e,i=holder['e'],holder['i']
        if not torch.equal(args[0],torch.tensor(items[e][0]['body'][i:i+1],dtype=torch.float32)):raise ValueError('Receiver body changed')
        return value+port(native[e][i:i+1],valid[e][i:i+1])
    handles=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(spatial),model.body_feedback.register_forward_hook(body)]
    started=time.monotonic();reports=[];out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    try:
        with np.load(root/'predictions.npz',allow_pickle=False) as reference,torch.no_grad():
            for e,(d,c,_) in enumerate(items):
                holder['e']=e;t=d['decision_time'];scene=torch.tensor(d['scene'],dtype=torch.float32)
                windows=prior_result['comparisons'][e]['windows'];starts={w['start'] for w in windows};states={};saved={};normal=[]
                def step(i,j=None,neutral=False):
                    j=i if j is None else j;holder.update(i=i,extra=extras[e][j:j+1],neutral=neutral)
                    _,mu=model.sensory_decision(decision_time=float(t[i]),racing_goals=goals,scene_features=scene[j:j+1],scene_valid=views,body_context=context(d,c,i))
                    return head(mu)[0].numpy().copy()
                model.reset_episode();optical.projection.weight.copy_(new_weight)
                for i in range(len(t)):
                    if i in starts:states[i]=snapshot(model)
                    normal.append(step(i))
                    if i%1000==0:write_json(run/'progress.json',dict(phase='continuous_fixed_replay',episode=e,event=i,total=len(t),elapsed_seconds=time.monotonic()-started))
                normal=np.asarray(normal);error=float(abs(normal-reference[f'prediction_{e}']).max())
                if error>2e-6:raise ValueError('Independent full neural replay differs from fixed fit predictions')
                saved['normal']=normal;window_reports=[]
                for wi,w in enumerate(windows):
                    ids=np.arange(w['start'],w['event']+1)
                    mask=pre[e][ids]&(t[ids]>=w['time']-.8)&(t[ids]<=w['time']-.08)
                    paths=dict(normal=ids,inherited_optical=ids,spatial_neutral=ids)
                    if w['donor_time'] is not None:paths['opposite']=floor_indices(t,t[ids]-w['time']+w['donor_time'])
                    predictions={}
                    for name,donors in paths.items():
                        restore(model,states[w['start']]);optical.projection.weight.copy_(old_weight if name=='inherited_optical' else new_weight)
                        values=[step(int(i),int(j),name=='spatial_neutral') for i,j in zip(ids,donors)]
                        predictions[name]=np.asarray(values);saved[f'window_{wi}_{name}']=predictions[name];saved[f'window_{wi}_{name}_indices']=donors
                    replay_error=float(abs(predictions['normal']-normal[ids]).max())
                    if replay_error>2e-6:raise ValueError('Restored normal window differs')
                    signed=float(np.mean((predictions['normal'][mask,0]-predictions['opposite'][mask,0])*w['direction'])) if 'opposite' in predictions else None
                    rmse={name:float(np.sqrt(np.mean((value[mask,0]-d['labels'][ids[mask],0])**2))) for name,value in predictions.items() if name!='opposite'}
                    window_reports.append(dict(event=w['event'],start=w['start'],direction=w['direction'],donor_event=w['donor_event'],
                        supervised_pre_count=int(mask.sum()),normal_replay_max_error=replay_error,signed_normal_minus_opposite=signed,original_label_rmse=rmse,
                        counterfactual_action_truth=False,same_initial_state_and_receiver_body=True))
                    saved[f'window_{wi}_receiver_indices']=ids;saved[f'window_{wi}_pre_mask']=mask
                optical.projection.weight.copy_(new_weight)
                file=run/f'episode-{e}.npz';np.savez_compressed(file,**saved)
                reports.append(dict(episode=e,model_events=len(t),full_replay_max_error=error,windows=window_reports,predictions_sha256=sha256(file)))
                write_json(run/'completed-episodes.json',dict(comparisons=reports))
        signed=[w['signed_normal_minus_opposite'] for report in reports for w in report['windows'] if w['signed_normal_minus_opposite'] is not None]
        out.update(status='completed_native_visual_anticipation_interventions',comparisons=reports,
            expected_direction_windows=sum(x>0 for x in signed),paired_windows=len(signed),
            termination_reason='Three independently reproduced full neural trajectories and same-state/body sustained visual feature interventions completed; no actor qualification')
    except Exception as error:out.update(error=type(error).__name__,reason=str(error),comparisons=reports,termination_reason='Explicit replay/intervention invariant failure')
    finally:
        for h in handles:h.remove()
    out['elapsed_seconds']=time.monotonic()-started;write_json(run/'result.json',out)
    if out['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
