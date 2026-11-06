"""Exact current initialization, five-group continuation and actual new-port adjoint."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.dual_reference_experiment import DualReferenceExperiment
from flybrain_drive.dual_reference_recording_objective import dual_reference_recording_losses
from scripts.sac_steering_experiment import restore
ROOT=Path('runs/dual-reference-learning-20261105')

def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=DualReferenceExperiment(p);source=Path(p['direct_parent_evaluation']);states=torch.load(source/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);out=begin_run(ROOT/'validation-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan-01.json'),seed=None,actor_qualified=False,vehicle_control_sent=False));parameters=e.train_parameters()
 try:
  optimizer=e.make_optimizer();old_optimizer=torch.load(p['direct_parent_optimizer'],map_location='cpu',weights_only=True)
  for j,w in enumerate(parameters[:4]):
   prior=old_optimizer['state'][old_optimizer['param_groups'][j]['params'][0]]
   for key in ['step','exp_avg','exp_avg_sq']:assert torch.equal(optimizer.state[w][key],prior[key])
  e.model.reset_episode();e.holder.update(mode='real',episode=0);prediction=np.asarray([e.step(i,False).numpy() for i in range(1024)])
  with np.load(source/'predictions.npz') as z:maximum=float(abs(prediction-z['prediction_0'][:1024]).max())
  assert maximum==0;e.verify_fixed();task=p['physical_tasks'][0];begin,end=task['begin'],task['end'];prefix=states[0,begin];d=e.items[0][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any()
  def loss():
   restore(e.model,prefix);v=torch.stack([e.step(i,True) for i in range(begin,end)]);return sum(dual_reference_recording_losses(v,torch.tensor(e.anchors[0][begin:end]),torch.tensor(e.parent_anchors[0][begin:end]),target,valid,torch.tensor(e.pre[0][begin:end]),anchor_weight=e.plan['real_anchor_weight'],maximum_ratio=e.plan['actual_recording_preservation_maximum_ratio'],protection_weight=e.plan.get('protection_weight',1.)))
  loss().backward();norms=[0. if w.grad is None else float(w.grad.norm()) for w in parameters];assert all(v>0 for v in norms[:4]) and parameters[4].grad is None;w=parameters[0];initial=w.detach().clone();direction=w.grad/w.grad.norm();analytic=norms[0];fd=[]
  for step in [.0002,.0001,.00002]:
   values=[]
   for sign in [-1,1]:
    with torch.no_grad():w.copy_(initial+sign*step*direction)
    values.append(float(loss().detach()))
   estimate=(values[1]-values[0])/(2*step);fd.append(dict(step=step,analytic=analytic,finite_difference=estimate,relative_error=abs(estimate-analytic)/analytic))
  with torch.no_grad():w.copy_(initial)
  assert min(v['relative_error'] for v in fd)<.05
  optimizer.zero_grad(set_to_none=True);loss().backward();old=[v.detach().clone() for v in parameters];optimizer.step();e.project_after_step(old);native=parameters[1].detach().clone();speed=[v.detach().clone() for v in parameters[2:4]];parts,_=e.synthetic(dict(source_slot=0,pair=0,temperature=e.sources[0]['plan']['temperature_c'][0],group=0),'episode_full');optimizer.zero_grad(set_to_none=True);sum(parts).backward();native_none=parameters[1].grad is None;policy_none=all(v.grad is None for v in parameters[2:4]);assert parameters[4].grad is None;old=[v.detach().clone() for v in parameters];optimizer.step();e.project_after_step(old);nc=float(abs(parameters[1].detach()-native).max());sc=max(float(abs(v.detach()-b).max()) for v,b in zip(parameters[2:4],speed));assert native_none and policy_none and nc==sc==0;e.verify_fixed();write_json(out/'result.json',dict(status='completed_dual_reference_validation',neutral_events=1024,neutral_maximum_prediction_difference=maximum,all_four_existing_adam_states_exact=True,new_gain_initial_optimizer_step=0,gradient_norms=norms,motion_gradient_finite_differences=fd,gradient_validated=True,synthetic_native_grad_is_none=native_none,synthetic_native_optimizer_change=nc,synthetic_policy_grad_is_none=policy_none,synthetic_policy_optimizer_change=sc,direct_gain_fixed_zero=True,disposable_optimizer_updates=2,actor_qualified=False,vehicle_control_sent=False))
 finally:e.close()

if __name__=='__main__':main()
