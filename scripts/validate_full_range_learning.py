"""Warm whole-history initialization and real registration/native gradients before fitting."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.full_range_port_experiment import FullRangePortExperiment
from scripts.sac_steering_experiment import restore

ROOT=Path('runs/full-range-port-learning-20261104')
def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');warm=[read(ROOT/f'initial-episode{e}-01/result.json') for e in [0,1]];assert all(r['status']=='completed_continuous_bilateral_registration_initialization' for r in warm);e=FullRangePortExperiment(plan,train_body=True);task=plan['physical_tasks'][0];begin,end=task['begin'],task['end'];states=torch.load(ROOT/'initial-episode0-01/prefix-states.pt',map_location='cpu',weights_only=True);prefix=states[begin];d=e.items[0][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any();e.holder.update(mode='real',episode=0);out=begin_run(ROOT/'validation-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan-01.json'),seed=None,warm_result_sha256=[sha256(ROOT/f'initial-episode{i}-01/result.json') for i in [0,1]],actor_qualified=False,vehicle_control_sent=False));parameters=e.train_parameters()
 try:
  def loss():
   restore(e.model,prefix);p=torch.stack([e.step(i,True) for i in range(begin,end)]);return (((p-target).square()*p.new_tensor([20.,1.,4.])).sum(1)[valid]).mean()
  loss().backward();rows=[]
  for name,weight in zip(['visual_projection','visual_bias','gate','native'],parameters):
   grad=weight.grad.detach().clone();norm=float(grad.norm());assert norm>0;direction=grad/grad.norm();initial=weight.detach().clone();finite=[]
   for step in ([.005,.001,.0005] if name=='visual_bias' else [.0001,.00002]):
    values=[]
    for sign in [-1,1]:
     with torch.no_grad():weight.copy_(initial+sign*step*direction)
     values.append(float(loss().detach()))
    estimate=(values[1]-values[0])/(2*step);finite.append(dict(step=step,analytic=norm,finite_difference=estimate,relative_error=abs(estimate-norm)/norm))
   with torch.no_grad():weight.copy_(initial)
   rows.append(dict(parameter=name,gradient_norm=norm,finite_differences=finite))
  optimizer=torch.optim.Adam([{'params':[p],'lr':lr} for p,lr in zip(parameters,e.learning_rates())]);optimizer.zero_grad(set_to_none=True);loss().backward();old=[p.detach().clone() for p in parameters];optimizer.step();e.project_after_step(old);native_real=parameters[3].detach().clone();parts,_=e.synthetic(dict(source_slot=0,pair=0,temperature=e.sources[0]['plan']['temperature_c'][0],group=0),'episode_full');optimizer.zero_grad(set_to_none=True);sum(parts).backward();native_none=parameters[3].grad is None;old=[p.detach().clone() for p in parameters];optimizer.step();e.project_after_step(old);change=float((parameters[3].detach()-native_real).abs().max());assert native_none and change==0;e.verify_fixed();write_json(out/'result.json',dict(status='completed_full_range_learning_validation',gradient_validated=all(min(v['relative_error'] for v in r['finite_differences'])<.1 for r in rows),gradients=rows,synthetic_native_grad_is_none=native_none,synthetic_native_optimizer_change=change,warm_train_yaw_correct=sum(r['yaw_both_correct'] for r in warm),warm_train_speed_correct=sum(r['speed_both_correct'] for r in warm),warm_train_cases=30,disposable_optimizer_updates=2,actor_qualified=False,vehicle_control_sent=False));print(rows)
 finally:e.close()
if __name__=='__main__':main()
