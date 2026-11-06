"""Verify exact parent replay and real motion/native gradient before fitting."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.spatial_motion_dn_experiment import SpatialMotionDNExperiment
from scripts.sac_steering_experiment import restore
ROOT=Path('runs/spatial-motion-dn-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=SpatialMotionDNExperiment(p,train_body=True);source=Path(p['motion_parent_evaluation']);r=read(source/'result.json');assert sha256(source/'candidate-prefix-states.pt')==r['prefix_states_sha256'];states=torch.load(source/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);out=begin_run(ROOT/'validation-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan-01.json'),seed=None,optimizer_updates=0,source_prefix_sha256=r['prefix_states_sha256'],actor_qualified=False,vehicle_control_sent=False));parameters=e.train_parameters()
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0);pred=np.asarray([e.step(i,False).numpy() for i in range(1024)])
  with np.load(source/'predictions.npz') as z:maximum=float(abs(pred-z['prediction_0'][:1024]).max())
  assert maximum==0;e.verify_fixed();task=p['physical_tasks'][0];begin,end=task['begin'],task['end'];prefix=states[0,begin];d=e.items[0][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any()
  def loss():
   restore(e.model,prefix);v=torch.stack([e.step(i,True) for i in range(begin,end)]);return (((v-target).square()*v.new_tensor([20.,1.,4.])).sum(1)[valid]).mean()
  loss().backward();rows=[]
  for name,w in zip(['motion_projection','native'],parameters):
   grad=w.grad.detach().clone();norm=float(grad.norm());assert norm>0;direction=grad/grad.norm();initial=w.detach().clone();values=[]
   for step in [.0001,.00002]:
    losses=[]
    for sign in [-1,1]:
     with torch.no_grad():w.copy_(initial+sign*step*direction)
     losses.append(float(loss().detach()))
    fd=(losses[1]-losses[0])/(2*step);values.append(dict(step=step,analytic=norm,finite_difference=fd,relative_error=abs(fd-norm)/norm))
   with torch.no_grad():w.copy_(initial)
   rows.append(dict(parameter=name,gradient_norm=norm,finite_differences=values))
  optimizer=torch.optim.Adam([{'params':[w],'lr':lr} for w,lr in zip(parameters,e.learning_rates())]);optimizer.zero_grad(set_to_none=True);loss().backward();old=[w.detach().clone() for w in parameters];optimizer.step();e.project_after_step(old);native=parameters[1].detach().clone();parts,_=e.synthetic(dict(source_slot=0,pair=0,temperature=e.sources[0]['plan']['temperature_c'][0],group=0),'episode_full');optimizer.zero_grad(set_to_none=True);sum(parts).backward();native_none=parameters[1].grad is None;old=[w.detach().clone() for w in parameters];optimizer.step();e.project_after_step(old);change=float(abs(parameters[1].detach()-native).max());assert native_none and change==0;e.verify_fixed();write_json(out/'result.json',dict(status='completed_motion_dn_learning_validation',neutral_events=1024,neutral_maximum_prediction_difference=maximum,gradients=rows,gradient_validated=all(min(v['relative_error'] for v in row['finite_differences'])<.1 for row in rows),synthetic_native_grad_is_none=native_none,synthetic_native_optimizer_change=change,disposable_optimizer_updates=2,actor_qualified=False,vehicle_control_sent=False));print(rows)
 finally:e.close()
if __name__=='__main__':main()
