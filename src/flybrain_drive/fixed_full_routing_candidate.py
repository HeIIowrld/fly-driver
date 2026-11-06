"""Hash-bound full-range input overlay for read-only research replay."""
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from .graph import sha256
from .sac_unified_student import read
from .full_range_dn_registration import FullRangeDNRegistration
from .visual_native_port_candidate import apply_visual_native_ports
from .motion_spatial_runtime import MotionSpatialDecisionRuntime


class FixedFullRoutingCurrent(nn.Module):
 def __init__(self,base,initial_gate):
  if initial_gate!=1.:raise ValueError('Fixed full routing requires gate one')
  super().__init__();self.base=deepcopy(base).eval().requires_grad_(False);self.registration=FullRangeDNRegistration(base.reference,base.scale,maximum=base.maximum)
  with torch.no_grad():self.registration.gate.fill_(initial_gate)
 def forward(self,features,parent):return self.registration(features,self.base(features,parent))


def apply_fixed_full_routing(checkpoint,components,calibration):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];p=m['plan']
 if r['status']!='completed_fixed_full_routing_fit' or sha256(path)!=r['checkpoint_sha256'] or any(sha256(p[key])!=p[key+'_sha256'] for key in ['parent_checkpoint','parent_port_checkpoint','fixed_decoder']):raise ValueError('Completed hash-bound full-range inputs required')
 cp=torch.load(path,map_location='cpu',weights_only=True)
 keys={'schema','arm','current_calibration','body_port','fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','initial_gate','minimum_gate','actor_qualified'}
 if set(cp)!=keys or cp['schema']!='flybrain.fixed_full_routing_ports.experimental.v1' or cp['arm'] not in ['visual_only','visual_native'] or cp['arm']!=m['arm'] or cp['actor_qualified'] is not False or any(cp[k]!=p[k] for k in keys-{'schema','arm','current_calibration','body_port','actor_qualified'}):raise ValueError('Exact unqualified full-range schema required')
 # Apply parent to private copies so a failed validation never mutates caller state.
 private=(components[0],components[1],components[2],components[3],deepcopy(components[4]),*components[5:]);base=deepcopy(calibration)
 changed,base,parent_receipt=apply_visual_native_ports(p['parent_port_checkpoint'],private,base);current=FixedFullRoutingCurrent(base,p['initial_gate']);body=changed[4];origin=body.projection.weight.detach().clone()
 variable={'registration.gate','registration.projection.weight','registration.projection.bias'}
 for module,key,allowed in [(current,'current_calibration',variable),(body,'body_port',{'projection.weight'})]:
  expected=module.state_dict();state=cp[key]
  if set(state)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k not in allowed and not torch.equal(v,expected[k]) for k,v in state.items()):raise ValueError('Frozen parent and exact finite normalization required')
 s=cp['current_calibration'];assert p['initial_gate']==p['minimum_gate']==1.;
 if not torch.equal(s['registration.gate'],torch.ones_like(s['registration.gate'])):raise ValueError('Fixed routing gate must equal one exactly')
 w=s['registration.projection.weight'];b=s['registration.projection.bias'];g=s['registration.gate'];n=cp['body_port']['projection.weight'];delta=n-origin
 if w.abs().max()>2+1e-7 or w.norm()>p['visual_total_norm']+1e-7 or b.abs().max()>2+1e-7 or b.norm()>p['bias_total_norm']+1e-7 or (g<p['minimum_gate']-1e-7).any() or (g>1).any() or delta.norm()>p['native_total_norm']+1e-7 or delta.abs().max()>p['native_entry_delta']+1e-7 or cp['arm']=='visual_only' and not torch.equal(n,origin):raise ValueError('Projected full-range/native bounds required')
 current.load_state_dict(cp['current_calibration']);body.load_state_dict(cp['body_port']);current.eval().requires_grad_(False);body.eval().requires_grad_(False)
 return changed,current,dict(parent_receipt,full_range_checkpoint_sha256=sha256(path),arm=cp['arm'],actor_qualified=False,vehicle_control_sent=False)


class FixedFullRoutingDecisionRuntime(MotionSpatialDecisionRuntime):
 def __init__(self,*components,current_calibration,**kwargs):
  if type(current_calibration) is not FixedFullRoutingCurrent or not torch.equal(current_calibration.registration.gate,torch.ones_like(current_calibration.registration.gate)) or any(t.device.type!='cpu' for t in [*current_calibration.parameters(),*current_calibration.buffers()]):raise ValueError('Explicit CPU full-range current required')
  self.current_calibration=current_calibration.eval().requires_grad_(False);super().__init__(*components,**kwargs)
 def _descending(self,module,args,value):
  steer,power=value;features=self.active['static'];return steer+self.current_calibration(features,self.descending(features)),power
