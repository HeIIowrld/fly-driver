"""Real TRAIN zero-overlay parity and gradient only; no new driver fitting."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
from flybrain_drive.spatial_motion_dn_overlay import SpatialMotionDNOverlay
from scripts.fixed_full_routing_experiment import FixedFullRoutingExperiment
from scripts.sac_steering_experiment import restore

ROOT=Path('runs/fixed-full-routing-learning-20261104')
class MotionDNProbe(FixedFullRoutingExperiment):
 def __init__(self,plan):
  super().__init__(plan,train_body=False);fit=ROOT/'visual_native-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port']);self.calibration.requires_grad_(False);parent=FixedFullRoutingCurrent(self.calibration.base,1.);parent.load_state_dict(self.calibration.state_dict());self.overlay=SpatialMotionDNOverlay(parent,self.motion.reference,self.motion.scale);self.parent_state={k:v.clone() for k,v in self.overlay.parent.state_dict().items()}
 def _descending(self,module,args,value):
  static,fine,_,_=self.inputs();steer,power=value;return steer+self.overlay(static,fine,self.dn(static)),power

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');e=MotionDNProbe(plan);evaluation=ROOT/'visual_native-evaluation-01';r=read(evaluation/'result.json');assert sha256(evaluation/'candidate-prefix-states.pt')==r['prefix_states_sha256'];states=torch.load(evaluation/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);out=begin_run(ROOT/'motion-dn-component-validation-01',dict(program_sha256=sha256(__file__),component_sha256=sha256('src/flybrain_drive/spatial_motion_dn_overlay.py'),seed=None,optimizer_updates=0,source_prefix_sha256=r['prefix_states_sha256'],actor_qualified=False,vehicle_control_sent=False));maximum=0.;rows=[]
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0);pred=np.asarray([e.step(i,False).numpy() for i in range(1024)])
  with np.load(evaluation/'predictions.npz') as z:maximum=float(abs(pred-z['prediction_0'][:1024]).max())
  assert maximum<2e-6
  for index in [0,3,5,15]:
   t=plan['physical_tasks'][index];episode,begin,end=t['episode'],t['begin'],t['end'];e.holder.clear();e.holder.update(mode='real',episode=episode);d=e.items[episode][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any()
   def loss():
    restore(e.model,states[episode,begin]);v=torch.stack([e.step(i,True) for i in range(begin,end)]);return (((v-target).square()*v.new_tensor([20.,1.,4.])).sum(1)[valid]).mean()
   e.overlay.zero_grad(set_to_none=True);loss().backward();w=e.overlay.motion_projection.weight;grad=w.grad.detach().clone();norm=float(grad.norm());assert norm>0;direction=grad/grad.norm();values=[]
   for step in [.0001,.00002]:
    losses=[]
    for sign in [-1,1]:
     with torch.no_grad():w.copy_(sign*step*direction)
     losses.append(float(loss().detach()))
    fd=(losses[1]-losses[0])/(2*step);values.append(dict(step=step,analytic=norm,finite_difference=fd,relative_error=abs(fd-norm)/norm))
   with torch.no_grad():w.zero_()
   rows.append(dict(task=t,projection_gradient_norm_by_port=grad.norm(dim=1).tolist(),finite_differences=values))
  assert all(torch.equal(v,e.parent_state[k]) for k,v in e.overlay.parent.state_dict().items());assert torch.count_nonzero(e.overlay.motion_projection.weight)==0
  write_json(out/'result.json',dict(status='completed_motion_dn_component_validation',neutral_events=1024,neutral_maximum_prediction_difference=maximum,gradient_reports=rows,gradient_validated=all(min(v['relative_error'] for v in row['finite_differences'])<.1 for row in rows),new_driver_trained=False,optimizer_updates=0,parent_preserved=True,actor_qualified=False,vehicle_control_sent=False))
 finally:e.close()
if __name__=='__main__':main()
