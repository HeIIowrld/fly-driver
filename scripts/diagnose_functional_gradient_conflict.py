"""TRAIN-only fixed-point objective conflict and full-history derivative check."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.functional_priority_experiment import Experiment

def cosine(a,b):
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b));return None if denominator<1e-24 else float(np.dot(a,b)/denominator)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1);torch.manual_seed(plan['seed']);e=Experiment(plan);previous=torch.load(plan['previous_checkpoint'],map_location='cpu',weights_only=True);assert sha256(plan['previous_checkpoint'])==plan['previous_checkpoint_sha256'] and previous['parent_candidate']==e.parent;tasks=[]
    for episode in (0,1):
        eligible=[t for t in e.real_tasks if t['episode']==episode and e.items[episode][0]['fit_mask'][t['begin']:t['end']].any()];assert len(eligible)>=6;tasks.extend(eligible[int(i)] for i in np.linspace(0,len(eligible)-1,6,dtype=int))
    synthetic=[dict(source_slot=s,pair=k,temperature=80.,group=3*(k//2)+1) for s in range(3) for k in (range(0,12,2) if s==0 else (2,8))];out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),parent_candidate=e.parent,program_sha256=sha256(__file__),selected_real_tasks=tasks,selected_synthetic_tasks=synthetic,actual_tasks_are_subset=True,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope=__doc__));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'backend-initialization.json',e.backend);arrays={};reports=[];start=time.perf_counter();before=gc.isenabled();gc.disable();result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False)
    def gradients(losses):return [torch.autograd.grad(loss,e.gain.value,retain_graph=j<len(losses)-1)[0].detach().numpy().copy() for j,loss in enumerate(losses)]
    try:
        for point,value in [('parent',torch.ones_like(e.gain.value)),('previous_gain',previous['temporal_gain']['value'])]:
            with torch.no_grad():e.gain.value.copy_(value)
            real=[];real_loss=[]
            for task in tasks:
                losses,_=e.real(task);real.append(gradients(losses));real_loss.append([float(v.detach()) for v in losses])
            real=np.asarray(real);arrays[f'{point}_real_gradients']=real;arrays[f'{point}_real_losses']=np.asarray(real_loss);syn={};parity=0.
            for mode in ('legacy','episode_full'):
                values=[];loss_values=[];pred=[]
                for task in synthetic:
                    losses,prediction=e.synthetic(task,mode);values.append(gradients(losses));loss_values.append([float(v.detach()) for v in losses]);pred.append(prediction.detach().numpy());write_json(out/'progress.json',dict(point=point,mode=mode,completed_synthetic=len(values),total=10,elapsed_seconds=time.perf_counter()-start))
                syn[mode]=np.asarray(values);arrays[f'{point}_{mode}_synthetic_gradients']=syn[mode];arrays[f'{point}_{mode}_synthetic_losses']=np.asarray(loss_values);arrays[f'{point}_{mode}_predictions']=np.asarray(pred)
            parity=float(abs(arrays[f'{point}_legacy_predictions']-arrays[f'{point}_episode_full_predictions']).max());assert parity<=2e-6;actual=real[:,0].mean(0);old=syn['legacy'][:,0].mean(0);full=syn['episode_full'][:,0].mean(0);cross=[cosine(a,b) for a in real[:,0] for b in syn['episode_full'][:,0]];cross=[v for v in cross if v is not None];losses,_=e.synthetic(synthetic[0],'episode_full');total_grad=gradients((sum(losses),))[0];baseline=e.gain.value.detach().clone();free=(baseline>.26)&(baseline<1.74);direction=torch.tensor(total_grad)*free
            if float(direction.norm())<1e-12:direction=free.to(torch.float32)
            direction=direction/direction.norm();fd=[];analytic=float(np.dot(total_grad,direction.numpy()));legacy_losses,_=e.synthetic(synthetic[0],'legacy');legacy_grad=gradients((sum(legacy_losses),))[0];legacy_direction=float(np.dot(legacy_grad,direction.numpy()));e.gain.value.requires_grad_(False)
            for epsilon in (.002,.0005):
                values=[]
                for sign in (1,-1):
                    with torch.no_grad():e.gain.value.copy_(baseline+sign*epsilon*direction)
                    losses,_=e.synthetic(synthetic[0],'episode_full');values.append(float(sum(losses)))
                numeric=(values[0]-values[1])/(2*epsilon);fd.append(dict(epsilon=epsilon,full_history_adjoint=analytic,legacy_partial_adjoint=legacy_direction,finite_difference=numeric,full_absolute_error=abs(numeric-analytic),legacy_absolute_error=abs(numeric-legacy_direction)))
            e.gain.value.requires_grad_(True)
            with torch.no_grad():e.gain.value.copy_(baseline)
            arrays[f'{point}_finite_difference_direction']=direction.numpy();reports.append(dict(point=point,real_supervised_mean_gradient_norm=float(np.linalg.norm(actual)),functional_full_mean_gradient_norm=float(np.linalg.norm(full)),supervised_vs_full_functional_cosine=cosine(actual,full),supervised_vs_legacy_functional_cosine=cosine(actual,old),real_functional_negative_pair_count=sum(v<0 for v in cross),real_functional_nonzero_pair_count=len(cross),legacy_vs_full_functional_cosine=cosine(old,full),same_numeric_trajectory_max_difference=parity,finite_difference_checks=fd))
        e.verify_fixed();np.savez_compressed(out/'gradients.npz',**arrays);result.update(status='completed_train_functional_gradient_conflict',reports=reports,real_tasks=len(tasks),synthetic_tasks=len(synthetic),gradient_points=2,gradients_sha256=sha256(out/'gradients.npz'),termination_reason='At same42 gain parameters, selected TRAIN real supervised/anchor/protection and synthetic direction/anchor gradients compared. Full common prefix/branch gradients checked against whole synthetic finite differences; legacy numerical trajectories retained for distinction. No optimizer/test feature/vehicle action; subset diagnostic, not entire TRAIN or physiology qualification.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:e.close();gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
