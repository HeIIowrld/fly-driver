"""Neutral parity, real neural gradients and native Adam isolation on disposable model."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.visual_native_port_experiment import VisualNativePortExperiment
from scripts.sac_steering_experiment import restore
from scripts.continuous_descending_experiment import detach_exact

ROOT=Path('runs/visual-native-port-learning-20261104')
def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');e=VisualNativePortExperiment(plan,train_body=True);out=begin_run(ROOT/'validation-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan-01.json'),seed=None,scope=__doc__,actor_qualified=False,vehicle_control_sent=False));sequence=[]
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0)
  for i in range(1024):
   sequence.append(e.step(i,False).numpy().copy())
   if (i+1)%32==0:detach_exact(e.model)
  with np.load('runs/multicorner-command-20261104/affine-evaluation-01/predictions.npz') as z:parity=float(abs(np.asarray(sequence)-z['prediction_0'][:1024]).max())
  assert parity<2e-6;task=plan['physical_tasks'][0];begin,end=task['begin'],task['end'];states=torch.load('runs/multicorner-command-20261104/episode0-probes-01/candidate-prefix-states.pt',map_location='cpu',weights_only=True);prefix=states[begin];target=torch.tensor(e.items[0][0]['labels'][begin:end]);e.holder.update(mode='real',episode=0)
  def loss():
   restore(e.model,prefix);pred=torch.stack([e.step(i,True) for i in range(begin,end)]);return ((pred-target).square()*pred.new_tensor([20.,1.,4.])).sum(1).mean()
  parameters=e.train_parameters();value=loss();value.backward();fd=[]
  for name,weight in zip(['visual','native'],parameters):
   gradient=weight.grad.detach().clone();norm=float(gradient.norm());assert norm>0;direction=gradient/gradient.norm();initial=weight.detach().clone();rows=[]
   for step in [.0001,.00002]:
    values=[]
    for sign in [-1,1]:
     with torch.no_grad():weight.copy_(initial+sign*step*direction)
     values.append(float(loss().detach()))
    estimate=(values[1]-values[0])/(2*step);rows.append(dict(step=step,analytic=norm,finite_difference=estimate,relative_error=abs(estimate-norm)/norm))
   with torch.no_grad():weight.copy_(initial)
   fd.append(dict(parameter=name,gradient_norm=norm,finite_differences=rows))
  optimizer=torch.optim.Adam(parameters,lr=.001);optimizer.zero_grad(set_to_none=True);loss().backward();old=[p.detach().clone() for p in parameters];optimizer.step();e.project_after_step(old);native_after_real=parameters[1].detach().clone();e.holder.pop('donor',None);parts,_=e.synthetic(dict(source_slot=0,pair=0,temperature=e.sources[0]['plan']['temperature_c'][0],group=0),'episode_full');optimizer.zero_grad(set_to_none=True);sum(parts).backward();synthetic_none=parameters[1].grad is None;old=[p.detach().clone() for p in parameters];optimizer.step();e.project_after_step(old);synthetic_change=float((parameters[1]-native_after_real).abs().max());assert synthetic_none and synthetic_change==0
  e.holder.update(mode='real',episode=0,counterfactual_images=True);restore(e.model,prefix);donor=plan['image_tasks'][0]['left_donor'];prediction=[]
  for offset,i in enumerate(range(begin,end)):e.holder['donor']=(0,donor+offset);prediction.append(e.step(i,True))
  optimizer.zero_grad(set_to_none=True);torch.stack(prediction)[:,0].square().mean().backward();image_none=parameters[1].grad is None;old=[p.detach().clone() for p in parameters];optimizer.step();e.project_after_step(old);image_change=float((parameters[1]-old[1]).abs().max());assert image_none and image_change==0;e.holder.pop('donor',None);e.verify_fixed();np.savez_compressed(out/'neutral.npz',prediction=np.asarray(sequence));result=dict(status='completed_visual_native_port_validation',neutral_events=1024,neutral_maximum_difference=parity,gradients=fd,gradient_validated=all(min(v['relative_error'] for v in row['finite_differences'])<.1 for row in fd),synthetic_native_grad_is_none=synthetic_none,synthetic_native_optimizer_change=synthetic_change,image_native_grad_is_none=image_none,image_native_optimizer_change=image_change,disposable_model_optimizer_updates=3,production_checkpoint_modified=False,actor_qualified=False,vehicle_control_sent=False);write_json(out/'result.json',result);print(result)
 finally:e.close()
if __name__=='__main__':main()
