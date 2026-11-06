"""Strict experimental motion-DN overlay for read-only replay; no actor authority."""
from pathlib import Path
from copy import deepcopy
import torch
from .graph import sha256
from .sac_unified_student import read
from .fixed_full_routing_candidate import apply_fixed_full_routing
from .spatial_motion_dn_overlay import SpatialMotionDNOverlay
from .motion_spatial_runtime import MotionSpatialDecisionRuntime


def apply_spatial_motion_dn(checkpoint,components,calibration,motion):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];p=m['plan']
 if r['status']!='completed_spatial_motion_dn_fit' or sha256(path)!=r['checkpoint_sha256'] or any(sha256(p[k])!=p[k+'_sha256'] for k in ['motion_parent_checkpoint','parent_checkpoint','parent_port_checkpoint','fixed_decoder']):raise ValueError('Completed hash-bound motion-DN inputs required')
 cp=torch.load(path,map_location='cpu',weights_only=True);keys={'schema','arm','current_calibration','body_port','fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','motion_parent_checkpoint_sha256','initial_gate','minimum_gate','actor_qualified'}
 if set(cp)!=keys or cp['schema']!='flybrain.spatial_motion_dn_learning.experimental.v1' or cp['arm'] not in ['visual_only','visual_native'] or cp['arm']!=m['arm'] or cp['actor_qualified'] is not False or (p['initial_gate']!=1. or p['minimum_gate']!=1.) or any(cp[k]!=p[k] for k in keys-{'schema','arm','current_calibration','body_port','actor_qualified'}):raise ValueError('Exact unqualified motion-DN schema required')
 private=(*components[:4],deepcopy(components[4]),*components[5:]);changed,parent,receipt=apply_fixed_full_routing(p['motion_parent_checkpoint'],private,deepcopy(calibration));current=SpatialMotionDNOverlay(parent,motion.reference,motion.scale);body=changed[4];origin=body.projection.weight.detach().clone()
 for module,key,allowed in [(current,'current_calibration',{'motion_projection.weight'}),(body,'body_port',{'projection.weight'})]:
  expected=module.state_dict();state=cp[key]
  if set(state)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k not in allowed and not torch.equal(v,expected[k]) for k,v in state.items()):raise ValueError('Frozen parent and exact motion/native normalization required')
 w=cp['current_calibration']['motion_projection.weight'];native=cp['body_port']['projection.weight'];delta=native-origin
 if w.abs().max()>2+1e-7 or w.norm()>p['visual_total_norm']+1e-7 or delta.norm()>p['native_total_norm']+1e-7 or delta.abs().max()>p['native_entry_delta']+1e-7 or cp['arm']=='visual_only' and not torch.equal(native,origin):raise ValueError('Declared motion/native bounds failed')
 current.load_state_dict(cp['current_calibration']);body.load_state_dict(cp['body_port']);current.eval().requires_grad_(False);body.eval().requires_grad_(False)
 return changed,current,dict(receipt,motion_dn_checkpoint_sha256=sha256(path),motion_parent_checkpoint_sha256=p['motion_parent_checkpoint_sha256'],arm=cp['arm'],actor_qualified=False,vehicle_control_sent=False)


class SpatialMotionDNDecisionRuntime(MotionSpatialDecisionRuntime):
 def __init__(self,*components,current_calibration,**kwargs):
  if type(current_calibration) is not SpatialMotionDNOverlay or any(t.device.type!='cpu' for t in [*current_calibration.parameters(),*current_calibration.buffers()]):raise ValueError('Explicit CPU spatial motion-DN overlay required')
  self.current_calibration=current_calibration.eval().requires_grad_(False);super().__init__(*components,**kwargs)
 def _descending(self,module,args,value):
  steer,power=value;features=self.active['static'];return steer+self.current_calibration(features,self.motion_active,self.descending(features)),power
