"""Zero speed-policy parity, all real gradients and fixture momentum exclusion."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.power_bounded_speed_experiment import PowerBoundedSpeedExperiment
from scripts.sac_steering_experiment import restore
ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=PowerBoundedSpeedExperiment(p,learn_speed=True);source=Path(p['bounded_parent_evaluation']);r=read(source/'result.json');assert sha256(source/'candidate-prefix-states.pt')==r['prefix_states_sha256'];states=torch.load(source/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);out=begin_run(ROOT/'validation-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan-01.json'),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));parameters=e.train_parameters()
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0);prediction=[];errors=[];powers=[]
  for i in range(1024):prediction.append(e.step(i,False).numpy());errors.append(float((e.last_command[:,1:2]-e.last_body_speed).abs().max()));powers.append(e.last_power.numpy().copy())
  with np.load(source/'predictions.npz') as z:maximum=float(abs(np.asarray(prediction)-z['prediction_0'][:1024]).max())
  assert maximum==0 and max(errors)==0;e.verify_fixed();t=p['physical_tasks'][0];begin,end=t['begin'],t['end'];prefix=states[0,begin];d=e.items[0][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any()
  def loss():
   restore(e.model,prefix);v=torch.stack([e.step(i,True) for i in range(begin,end)]);return (((v-target).square()*v.new_tensor([20.,1.,4.])).sum(1)[valid]).mean()
  loss().backward();rows=[]
  for name,w in zip(['motion','native','speed_weight','speed_bias'],parameters):
   grad=w.grad.detach().clone();norm=float(grad.norm());assert norm>0;direction=grad/grad.norm();initial=w.detach().clone();values=[]
   for step in ([.001,.0002,.0001] if name.startswith('speed') else [.0001,.00002]):
    losses=[]
    for sign in [-1,1]:
     with torch.no_grad():w.copy_(initial+sign*step*direction)
     losses.append(float(loss().detach()))
    fd=(losses[1]-losses[0])/(2*step);values.append(dict(step=step,analytic=norm,finite_difference=fd,relative_error=abs(fd-norm)/norm))
   with torch.no_grad():w.copy_(initial)
   rows.append(dict(parameter=name,gradient_norm=norm,finite_differences=values))
  optimizer=e.make_optimizer();initial_steps=[int(optimizer.state[w]['step']) for w in parameters[:2]];optimizer.zero_grad(set_to_none=True);loss().backward();old=[w.detach().clone() for w in parameters];optimizer.step();e.project_after_step(old);native=parameters[1].detach().clone();speed=[w.detach().clone() for w in parameters[2:]];parts,_=e.synthetic(dict(source_slot=0,pair=0,temperature=e.sources[0]['plan']['temperature_c'][0],group=0),'episode_full');optimizer.zero_grad(set_to_none=True);sum(parts).backward();native_none=parameters[1].grad is None;policy_none=all(w.grad is None for w in parameters[2:]);old=[w.detach().clone() for w in parameters];optimizer.step();e.project_after_step(old);nc=float(abs(parameters[1].detach()-native).max());sc=max(float(abs(w.detach()-before).max()) for w,before in zip(parameters[2:],speed));assert native_none and policy_none and nc==sc==0;e.verify_fixed();write_json(out/'result.json',dict(status='completed_power_bounded_speed_validation',neutral_events=1024,neutral_maximum_prediction_difference=maximum,neutral_maximum_speed_error=max(errors),neutral_power_variation=float(np.ptp(np.asarray(powers),axis=0).max()),gradients=rows,gradient_validated=all(min(v['relative_error'] for v in row['finite_differences'])<.1 for row in rows),initial_optimizer_steps=initial_steps,synthetic_native_grad_is_none=native_none,synthetic_native_optimizer_change=nc,synthetic_policy_grad_is_none=policy_none,synthetic_policy_optimizer_change=sc,disposable_optimizer_updates=2,actor_qualified=False,vehicle_control_sent=False));print(rows)
 finally:e.close()
if __name__=='__main__':main()
