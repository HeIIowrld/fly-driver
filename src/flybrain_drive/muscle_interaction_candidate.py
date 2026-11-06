"""Strict offline loading of a muscle-only experimental decoder."""
from pathlib import Path
import torch
from .graph import sha256
from .sac_unified_student import read
from .muscle_interaction_vehicle_readout import MuscleInteractionVehicleReadout

def load_muscle_interaction_head(checkpoint):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];plan=m['protocol'];parent_path=Path(plan['parent_checkpoint'])
 if r['status']!='completed_multicorner_constrained_decoder_fit' or sha256(path)!=r['checkpoint_sha256'] or sha256(parent_path)!=plan['parent_checkpoint_sha256']:raise ValueError('Completed hash-bound experimental decoder and original parent required')
 cp=torch.load(path,map_location='cpu',weights_only=True);degree=1 if r['variant']=='affine' else 2
 if set(cp)!={'schema','degree','readout','parent_checkpoint_sha256','actor_qualified'} or cp['schema']!='flybrain.muscle_interaction_decoder.experimental.v1' or type(cp['degree']) is not int or cp['degree']!=degree or cp['parent_checkpoint_sha256']!=plan['parent_checkpoint_sha256'] or cp['actor_qualified'] is not False:raise ValueError('Explicit finite muscle interaction schema and no actor authority required')
 parent=torch.load(parent_path,map_location='cpu',weights_only=True)['readout'];s=cp['readout'];head=MuscleInteractionVehicleReadout(parent,degree=degree,quadratic_reference=s['quadratic_reference'],quadratic_scale=s['quadratic_scale']);expected=head.state_dict();variable={'steering.weight','steering.bias','quadratic_reference','quadratic_scale'}
 if set(s)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k not in variable and not torch.equal(v,expected[k]) for k,v in s.items()):raise ValueError('Exact unchanged original pedal/normalization tensors required')
 head.load_state_dict(s,strict=True)
 if head.steering.weight.abs().max()>plan['coefficient_bound']+1e-7 or head.steering.bias.abs().max()>plan['coefficient_bound']+1e-7:raise ValueError('Constrained coefficient bound exceeded')
 head.eval().requires_grad_(False);return head,dict(checkpoint_sha256=sha256(path),degree=degree,parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],actor_qualified=False,vehicle_control_sent=False)
