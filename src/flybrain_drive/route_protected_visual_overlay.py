"""Research overlay limited to existing pooled visual/motion input weights."""
from pathlib import Path
import torch
from .audited_motion_spatial_candidate import require_frozen_motion_parent
from .functional_motion_overlay import exact_model
from .graph import sha256
from .sac_unified_student import read


def require_pooled_only_model(candidate,parent,bound):
    if not 0<bound<=.002 or set(candidate)!=set(parent):raise ValueError('Bounded same-schema pooled visual model required')
    fixed={k:v for k,v in parent.items() if k!='visual.weight'};other={k:v for k,v in candidate.items() if k!='visual.weight'}
    if not exact_model(fixed,other):raise ValueError('Nonvisual graph/body/bias/other model state changed')
    value,old=candidate['visual.weight'],parent['visual.weight']
    if value.shape!=old.shape or value.dtype!=old.dtype or not torch.isfinite(value).all() or not torch.equal(value[:,:12],old[:,:12]) or float(abs(value-old).max())>bound+1e-7:raise ValueError('Pooled sensory weights exceed allowed columns/bound')


def load_route_protected_overlay(path,components,motion,receipt):
    path=Path(path);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];cp=torch.load(path,map_location='cpu',weights_only=True);plan=m['plan']
    if r['status']!='completed_route_protected_visual_fit' or sha256(path)!=r['checkpoint_sha256'] or cp['schema']!='flybrain.route_protected_visual.v1' or cp['actor_qualified'] is not False or cp['parent_candidate']!=receipt or m['parent_candidate']!=receipt or cp['optimizer_updates']!=r['optimizer_updates']:raise ValueError('Exact unqualified route-protected visual fit required')
    require_frozen_motion_parent(cp,components);require_pooled_only_model(cp['model'],components[0].state_dict(),plan['maximum_pooled_weight_change'])
    state=cp['motion_port'];old=motion.state_dict()
    if set(state)!=set(old) or any(v.shape!=old[k].shape or v.dtype!=old[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Motion schema changed')
    if any(not torch.equal(state[k],v) for k,v in old.items() if k!='projection.weight') or not 0<plan['maximum_motion_weight_change']<=.002 or float(abs(state['projection.weight']-old['projection.weight']).max())>plan['maximum_motion_weight_change']+1e-7:raise ValueError('Motion normalization/change bound differs')
    components[0].load_state_dict(cp['model'],strict=True);motion.load_state_dict(state,strict=True);motion.eval().requires_grad_(False)
    return dict(receipt,route_protected_checkpoint_sha256=sha256(path),route_scope='Research pooled/motion visual correction only; no live/actuator authority',actor_qualified=False,vehicle_control_sent=False)
