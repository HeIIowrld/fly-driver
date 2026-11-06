"""Strict experimental port overlay; inherits no vehicle-control authority."""
from pathlib import Path
import torch
from .graph import sha256
from .sac_unified_student import read
from .muscle_interaction_candidate import load_muscle_interaction_head

def apply_visual_native_ports(checkpoint,components,calibration):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];plan=m['plan']
 if r['status']!='completed_visual_native_port_fit' or sha256(path)!=r['checkpoint_sha256'] or sha256(plan['fixed_decoder'])!=plan['fixed_decoder_sha256'] or sha256(plan['parent_checkpoint'])!=plan['parent_checkpoint_sha256']:raise ValueError('Completed exact experimental inputs required')
 cp=torch.load(path,map_location='cpu',weights_only=True)
 if set(cp)!={'schema','arm','current_calibration','body_port','fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','actor_qualified'} or cp['schema']!='flybrain.visual_native_ports.experimental.v1' or cp['arm'] not in ['visual_only','visual_native'] or cp['arm']!=m['arm'] or cp['fixed_decoder']!=plan['fixed_decoder'] or cp['fixed_decoder_sha256']!=plan['fixed_decoder_sha256'] or cp['parent_checkpoint_sha256']!=plan['parent_checkpoint_sha256'] or cp['actor_qualified'] is not False:raise ValueError('Unqualified port schema and fixed decoder required')
 body=components[4];origins=[calibration.projection.weight.detach().clone(),body.projection.weight.detach().clone()]
 for module,key in [(calibration,'current_calibration'),(body,'body_port')]:
  expected=module.state_dict();state=cp[key]
  if set(state)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k!='projection.weight' and not torch.equal(v,expected[k]) for k,v in state.items()):raise ValueError('Exact original sensory normalization and finite shapes required')
 current=cp['current_calibration']['projection.weight'];native=cp['body_port']['projection.weight'];vd=current-origins[0];nd=native-origins[1]
 if current.abs().max()>plan['visual_absolute_bound']+1e-7 or vd.norm()>plan['visual_total_norm']+1e-7 or nd.norm()>plan['native_total_norm']+1e-7 or nd.abs().max()>plan['native_entry_delta']+1e-7 or cp['arm']=='visual_only' and not torch.equal(native,origins[1]):raise ValueError('Declared port limits/comparator preservation failed')
 head,receipt=load_muscle_interaction_head(plan['fixed_decoder']);calibration.load_state_dict(cp['current_calibration'],strict=True);body.load_state_dict(cp['body_port'],strict=True)
 for module in [head,calibration,body]:module.eval().requires_grad_(False)
 return (components[0],head,*components[2:]),calibration,dict(receipt,port_checkpoint_sha256=sha256(path),arm=cp['arm'],actor_qualified=False,vehicle_control_sent=False)
