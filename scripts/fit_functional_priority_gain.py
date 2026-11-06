"""Matched full-history temporal gain fits: command target or functional-first protection."""
import argparse,gc,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.functional_priority_experiment import Experiment


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--arm',choices=['supervised','functional_priority'],required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed']);assert sha256(plan['reserved_test_plan'])==plan['reserved_test_plan_sha256'] and sha256(plan['gradient_validation_result'])==plan['gradient_validation_result_sha256'] and read(plan['gradient_validation_result'])['all_largest_step_relative_errors_below_05'];e=Experiment(plan);actual=list(e.real_tasks);synthetic=[dict(source_slot=s,pair=k,temperature=t,group=3*(k//2)+j) for s in range(3) for k in range(0,12,2) for j,t in enumerate(e.sources[0]['plan']['temperature_c'])];rng.shuffle(actual);rng.shuffle(synthetic);pending=dict(zip(map(int,np.linspace(0,len(actual)-1,len(synthetic),dtype=int)),synthetic));schedule=[]
    for j,task in enumerate(actual):
        schedule.append(('real',task))
        if j in pending:schedule.append(('synthetic',pending[j]))
    assert len(actual)==93 and len(synthetic)==54 and len(schedule)==147;out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),parent_candidate=e.parent,entries=e.entries,real_windows=e.real_tasks,train_episodes=[0,1],development_episode_used_for_fit=False,seed=plan['seed'],actor_qualified=False,vehicle_control_sent=False,synthetic_gradient_mode='episode_full',scope=__doc__));write_json(out/'training-schedule.json',schedule);write_json(out/'backend-initialization.json',e.backend)
    for path in plan['source_sha256']:
        target=out/'executed-source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    optimizer=torch.optim.Adam([e.gain.value],lr=plan['learning_rate']);start=time.perf_counter();before=gc.isenabled();gc.disable();records=[];gain_before=[];gain_after=[];gradients=[];result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    try:
        for kind,task in schedule:
            if kind=='real':
                parts,_=e.real(task);supervised,anchor,protection=parts;loss=anchor+(supervised if a.arm=='supervised' else protection);names=['supervision','anchor','protection']
            else:
                parts,_=e.synthetic(task,'episode_full');loss=sum(parts);names=['direction','anchor']
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_([e.gain.value],1.);assert torch.isfinite(norm);gain_before.append(e.gain.value.detach().numpy().copy());gradients.append(e.gain.value.grad.detach().numpy().copy());optimizer.step();e.gain.project();gain_after.append(e.gain.value.detach().numpy().copy());records.append(dict(kind=kind,task=task,selected_objective=a.arm if kind=='real' else 'direction_plus_anchor',loss=float(loss.detach()),parts={name:float(value.detach()) for name,value in zip(names,parts)},gradient_norm=float(norm)));write_json(out/'progress.json',dict(optimizer_updates=len(records),total=len(schedule),elapsed_seconds=time.perf_counter()-start))
        e.verify_fixed();cp=out/'temporal-gain.pt';model,head,dn,optical,body_port,_=e.components;torch.save(dict(schema='flybrain.temporal_gain.v1',temporal_gain=e.gain.state_dict(),parent_candidate=e.parent,model=model.state_dict(),motion_port=e.motion.state_dict(),readout=head.state_dict(),adapter=dn.state_dict(),optical=optical.state_dict(),body_port=body_port.state_dict(),optimizer_updates=len(records),actor_qualified=False,training_regime=a.arm,synthetic_gradient_mode='episode_full'),cp);np.savez_compressed(out/'optimizer-trajectory.npz',gain_before=np.asarray(gain_before),gain_after=np.asarray(gain_after),applied_gradients=np.asarray(gradients));write_json(out/'training-records.json',records);result.update(status='completed_temporal_gain_fit',arm=a.arm,optimizer_updates=len(records),real_updates=93,synthetic_updates=54,parent_full_train_max_differences=[0.,0.],parent_prefix_reused=True,parent_prefix_source=plan['prefix_run'],parent_prefix_states_sha256=sha256(e.prefix_file),checkpoint_sha256=sha256(cp),training_records_sha256=sha256(out/'training-records.json'),optimizer_trajectory_sha256=sha256(out/'optimizer-trajectory.npz'),maximum_gain_change=float(abs(e.gain.value-1).max()),maximum_gradient_norm=max(v['gradient_norm'] for v in records),synthetic_gradient_mode='episode_full',original_chemical_graph_body_readout_exact=True,reserved_test_features_read=False,termination_reason='Matched147-update full synthetic history/prefix gradient temporal gains only; real command fitting or protection-only comparator. Numerical parent real cuts explicit and unchanged, continuous updated trajectories and fresh reserved test evaluated separately. No actor/vehicle authority.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),optimizer_updates=len(records))
    finally:e.close();gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
