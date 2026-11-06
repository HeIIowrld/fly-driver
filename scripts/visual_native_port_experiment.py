"""Train existing visual/native input ports under one fixed signed muscle decoder."""
from pathlib import Path
from copy import deepcopy
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.functional_motion_overlay import exact_model
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import context

@torch.no_grad()
def bounded_step(parameter,old,origin,*,step_norm,total_norm,entry_delta=None,absolute=None):
 delta=parameter-old;parameter.copy_(old+delta*(step_norm/delta.norm().clamp_min(1e-30)).clamp_max(1.));delta=parameter-origin;parameter.copy_(origin+delta*(total_norm/delta.norm().clamp_min(1e-30)).clamp_max(1.))
 if entry_delta is not None:parameter.copy_(origin+(parameter-origin).clamp(-entry_delta,entry_delta))
 if absolute is not None:parameter.clamp_(-absolute,absolute)
 if float((parameter-old).norm())>step_norm+1e-7 or float((parameter-origin).norm())>total_norm+1e-7:raise ValueError('Projected step/total norm exceeded')

class VisualNativePortExperiment(JointDNDecoderExperiment):
 def __init__(self,plan,*,train_body):
  super().__init__(read(plan['parent_fit_plan']),4,torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=True)['readout']);self.port_plan=plan;self.train_body=bool(train_body);parent=torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=True);self.calibration.load_state_dict(parent['current_calibration']);self.head,self.head_receipt=load_muscle_interaction_head(plan['fixed_decoder']);self.components=(self.model,self.head,*self.components[2:]);self.fixed_head={k:v.clone() for k,v in self.head.state_dict().items()};self.current_origin=self.calibration.projection.weight.detach().clone();self.native_origin=self.body_port.projection.weight.detach().clone();self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False);self.calibration.requires_grad_(True);self.body_port.projection.requires_grad_(self.train_body);self.last_muscles=None;self.hooks.append(self.head.register_forward_pre_hook(self._muscle));self.original_native_buffers={k:v.clone() for k,v in self.body_port.state_dict().items() if k!='projection.weight'}
 def _muscle(self,module,args):self.last_muscles=args[0]
 def _body(self,module,args,value):
  _,_,native,valid=self.inputs()
  port=self.baseline_native_port if self.holder.get('native_change_off',False) else self.body_port
  if self.holder['mode']=='synthetic' or self.holder.get('counterfactual_images',False):
   # Preserve the fixture's forward value but never train real native sensing
   # from its invented body values. Grad=None also suppresses Adam momentum.
   with torch.no_grad():delta=port(native,valid)
  else:delta=port(native,valid)
  return value+delta
 def _descending(self,module,args,value):
  static,_,_,_=self.inputs();steer,power=value;port=self.baseline_calibration if self.holder.get('visual_change_off',False) else self.calibration
  return steer+port(static,self.dn(static)),power
 def inputs(self):
  original=super().inputs();donor=self.holder.get('donor')
  if donor is None:return original
  episode,index=donor;j=int(self.maps[episode][index]);return self.extras[episode][index:index+1],self.features[episode][j:j+1],original[2],original[3]
 def step(self,i,learn):
  donor=self.holder.get('donor')
  if donor is None:return super().step(i,learn)
  self.holder['index']=i;d,c,_=self.items[self.holder['episode']];de,di=donor
  with torch.set_grad_enabled(learn):
   fn=self.model.sensory_training_decision if learn else self.model.sensory_decision;_,mu=fn(decision_time=float(d['decision_time'][i]),racing_goals=self.goals,scene_features=torch.tensor(self.items[de][0]['scene'][di:di+1]),scene_valid=self.views,body_context=context(d,c,i));return self.head(mu)[0]
 def train_parameters(self):return [self.calibration.projection.weight]+([self.body_port.projection.weight] if self.train_body else [])
 def project_after_step(self,old):
  p=self.port_plan;bounded_step(self.calibration.projection.weight,old[0],self.current_origin,step_norm=p['visual_step_norm'],total_norm=p['visual_total_norm'],absolute=p['visual_absolute_bound'])
  if self.train_body and self.body_port.projection.weight.grad is not None:bounded_step(self.body_port.projection.weight,old[1],self.native_origin,step_norm=p['native_step_norm'],total_norm=p['native_total_norm'],entry_delta=p['native_entry_delta'])
 def verify_fixed(self):
  for index,(module,prior) in enumerate(zip([*self.components[:5],self.motion],self.protected)):
   if index==1:
    if not exact_model(module.state_dict(),self.fixed_head):raise ValueError('Fixed signed decoder changed')
   elif index==4:
    state={k:v for k,v in module.state_dict().items() if k!='projection.weight'}
    if not exact_model(state,self.original_native_buffers):raise ValueError('Native normalization changed')
   elif not exact_model(module.state_dict(),prior):raise ValueError('Protected chemical/visual/legacy-body component changed '+str(index))
  if not torch.equal(self.gain.value,torch.ones_like(self.gain.value)):raise ValueError('Temporal gain changed')
  if not self.train_body and not torch.equal(self.body_port.projection.weight,self.native_origin):raise ValueError('Visual-only comparator native weights changed')
  p=self.port_plan
  for weight,origin,limit in [(self.calibration.projection.weight,self.current_origin,p['visual_total_norm']),(self.body_port.projection.weight,self.native_origin,p['native_total_norm'])]:
   if not torch.isfinite(weight).all() or float((weight-origin).norm())>limit+1e-7:raise ValueError('Finite bounded port calibration required')
