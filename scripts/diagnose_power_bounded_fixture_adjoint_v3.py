"""Compare stop-all versus frozen-parameter fixture gradients on one TRAIN image fork."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.fixture_speed_adjoint import frozen_parameter_speed_command
from scripts.power_bounded_speed_experiment import PowerBoundedSpeedExperiment
from scripts.sac_steering_experiment import restore
ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=PowerBoundedSpeedExperiment(p,learn_speed=True);fit=ROOT/'speed_policy-fit-01';cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);ev=ROOT/'speed_policy-evaluation-01';er=read(ev/'result.json');assert sha256(ev/'candidate-prefix-states.pt')==er['prefix_states_sha256'];states=torch.load(ev/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);task=p['image_tasks'][0];begin,end=task['begin'],task['end'];prefix=states[task['episode'],begin];out=begin_run(ROOT/'fixture-visual-adjoint-03',dict(program_sha256=sha256(__file__),helper_sha256=sha256('src/flybrain_drive/fixture_speed_adjoint.py'),checkpoint_sha256=sha256(fit/'ports.pt'),task=task,optimizer_updates=0,scope='One existing TRAIN retimed image pair, same numeric prefix/body/images/clocks and nonzero speed coefficients. Exact registered image-exchange update objective with the neutral target held at base weights across finite differences, as in one optimizer update; diagnose the fixture visual adjoint; no new sensor truth, physics or model selection.',actor_qualified=False,vehicle_control_sent=False));saved={};rows={}
 def override(module,args,value):
  if e.holder.get('diagnostic_adjoint')=='frozen_parameters':return frozen_parameter_speed_command(module,*args)
  return value
 e.hooks.append(e.speed_policy.register_forward_hook(override))
 def loss(mode,learn):
  branches=[]
  for donor in [task['left_donor'],task['right_donor']]:
   restore(e.model,prefix);e.holder.clear();e.holder.update(mode='real',episode=task['episode'],counterfactual_images=True,diagnostic_adjoint=mode);seq=[]
   for offset,i in enumerate(range(begin,end)):e.holder['donor']=(task['episode'],donor+offset);seq.append(e.step(i,learn))
   branches.append(torch.stack(seq))
  values=torch.stack(branches);neutral=neutral_anchor;left,right=values;pair_margin=-(left[:,0]-right[:,0])/2;direction=p['image_direction_weight']*torch.relu(p['image_target_margin']-pair_margin).square().mean();centre=p['image_anchor_weight']*((.5*(left+right)-neutral).square()*left.new_tensor([4.,1.,1.])).sum(1).mean();return direction+centre,values
 restore(e.model,prefix);e.holder.clear();e.holder.update(mode='real',episode=task['episode'],counterfactual_images=True,diagnostic_adjoint='stop_all');neutral_anchor=torch.stack([e.step(i,False) for i in range(begin,end)]).detach().clone()
 try:
  saved['frozen_neutral_anchor']=neutral_anchor.numpy();gradients={}
  for mode in ['stop_all','frozen_parameters']:
   for v in e.train_parameters():v.grad=None
   value,pred=loss(mode,True);value.backward();g=e.calibration.projection.weight.grad.detach().clone();assert all(v.grad is None for v in [*e.speed_policy.parameters(),e.body_port.projection.weight]);gradients[mode]=g;saved[mode+'_gradient']=g.numpy();saved[mode+'_prediction']=pred.detach().numpy();rows[mode]=dict(registered_image_loss=float(value.detach()),visual_gradient_norm=float(g.norm()),speed_and_native_grad_is_none=True)
  parity=float(abs(saved['stop_all_prediction']-saved['frozen_parameters_prediction']).max());assert parity==0;difference=gradients['frozen_parameters']-gradients['stop_all'];assert difference.norm()>0;direction=difference/difference.norm();w=e.calibration.projection.weight;initial=w.detach().clone();fd=[]
  for step in [.0002,.0001,.00002]:
   values=[]
   for sign in [-1,1]:
    with torch.no_grad():w.copy_(initial+sign*step*direction)
    value,_=loss('stop_all',False);values.append(float(value))
   finite=(values[1]-values[0])/(2*step);fd.append(dict(step=step,finite_difference=finite,stop_all_analytic=float((gradients['stop_all']*direction).sum()),frozen_parameters_analytic=float((gradients['frozen_parameters']*direction).sum())))
  with torch.no_grad():w.copy_(initial)
  e.verify_fixed();np.savez_compressed(out/'values.npz',**saved);write_json(out/'result.json',dict(status='completed_fixture_visual_adjoint_diagnostic',reports=rows,maximum_numeric_prediction_difference=parity,gradient_difference_norm=float(difference.norm()),directional_derivatives=fd,values_sha256=sha256(out/'values.npz'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));print(rows,fd)
 finally:e.close()
if __name__=='__main__':main()
