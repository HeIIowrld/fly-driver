"""Hash-bound experimental current/readout overlay; never grants actor authority."""
from pathlib import Path
import torch
from .evidence import write_json
from .graph import sha256
from .sac_unified_student import read
from .visual_dn_current_calibration import VisualDNCurrentCalibration,PowerMonotoneVehicleReadout
from .motion_spatial_runtime import MotionSpatialDecisionRuntime

def load_joint_dn_decoder(checkpoint,components,parent):
 path=Path(checkpoint);result=read(path.parent/'result.json');manifest=read(path.parent/'manifest.json')['config'];plan=manifest['plan']
 if result['status']!='completed_joint_dn_decoder_fit' or sha256(path)!=result['checkpoint_sha256'] or sha256(manifest['plan']['candidate_plan'])!=plan['candidate_plan_sha256'] or manifest['parent_candidate']!=parent:raise ValueError('Completed hash-bound original parent experiment required')
 state=torch.load(path,map_location='cpu',weights_only=True)
 if set(state)!={'schema','readout','current_calibration','rank','parent_candidate','actor_qualified'} or state['schema']!='flybrain.joint_dn_decoder.experimental.v1' or type(state['rank']) is not int or state['rank'] not in [1,4] or state['rank']!=manifest['rank'] or state['parent_candidate']!=parent or state['actor_qualified'] is not False:raise ValueError('Explicit unqualified joint decoder schema required')
 original=components[1].state_dict();head=PowerMonotoneVehicleReadout(original,plan['minimum_power_weight']);dn=components[2];calibration=VisualDNCurrentCalibration(dn.reference,dn.scale,dn.orientations,rank=state['rank'],maximum=dn.maximum,maximum_logit_shift=plan['maximum_logit_shift'])
 for module,key,variable in [(head,'readout',{'projection.weight','projection.bias'}),(calibration,'current_calibration',{'projection.weight'})]:
  expected=module.state_dict();value=state[key]
  if set(value)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k not in variable and not torch.equal(v,expected[k]) for k,v in value.items()):raise ValueError('Exact original normalization and finite tensors required')
  module.load_state_dict(value,strict=True);module.eval().requires_grad_(False)
 w=head.projection.weight
 if w.abs().max()>10 or head.projection.bias.abs().max()>10 or (w[1,4:]<plan['minimum_power_weight']-1e-8).any() or (w[2,4:]>-plan['minimum_power_weight']+1e-8).any() or calibration.projection.weight.abs().max()>plan['maximum_current_weight']+1e-7:raise ValueError('Power signs and calibration bounds required')
 return (components[0],head,*components[2:]),calibration,dict(checkpoint_sha256=sha256(path),rank=state['rank'],actor_qualified=False,vehicle_control_sent=False)

class JointDNDecoderDecisionRuntime(MotionSpatialDecisionRuntime):
 def __init__(self,*components,current_calibration,**kwargs):
  if type(current_calibration) is not VisualDNCurrentCalibration or any(t.device.type!='cpu' for t in [*current_calibration.parameters(),*current_calibration.buffers()]):raise ValueError('Explicit CPU visual current calibration required')
  self.current_calibration=current_calibration.eval().requires_grad_(False);super().__init__(*components,**kwargs)
 def _descending(self,module,args,value):
  steer,power=value;features=self.active['static'];return steer+self.current_calibration(features,self.descending(features)),power
