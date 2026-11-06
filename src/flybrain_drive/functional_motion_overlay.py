"""Research-only functional overlay; no live candidate or actuator loader."""
from pathlib import Path
import torch
from .audited_motion_spatial_candidate import exact,require_frozen_motion_parent
from .graph import sha256
from .sac_unified_student import read


def exact_model(left,right):
    if set(left)!=set(right):return False
    for key,value in left.items():
        other=right[key]
        if value.layout!=other.layout or value.shape!=other.shape or value.dtype!=other.dtype:return False
        if value.layout==torch.strided:
            if not torch.equal(value,other):return False
        else:
            a,b=value.to_sparse_coo().coalesce(),other.to_sparse_coo().coalesce()
            if not torch.equal(a.indices(),b.indices()) or not torch.equal(a.values(),b.values()):return False
    return True


def load_functional_motion_overlay(path,components,motion,receipt):
    path=Path(path);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];cp=torch.load(path,map_location='cpu',weights_only=True)
    if (r['status']!='completed_controlled_motion_function_fit' or r['checkpoint_sha256']!=sha256(path) or cp['schema']!='flybrain.functional_motion_port.v1' or cp['actor_qualified'] is not False or cp['parent_candidate']!=receipt or m['parent_candidate']!=receipt or cp['optimizer_updates']!=r['optimizer_updates'] or cp['maximum_weight_change']!=m['plan']['maximum_weight_change'] or not 0<cp['maximum_weight_change']<=.002):raise ValueError('Exact completed unqualified functional overlay required')
    require_frozen_motion_parent(cp,components)
    if not exact_model(cp['model'],components[0].state_dict()):raise ValueError('Functional overlay changed original model')
    state=cp['motion_port'];original=motion.state_dict()
    if set(state)!=set(original) or any(v.dtype!=original[k].dtype or v.shape!=original[k].shape or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Functional motion state schema differs')
    if any(not torch.equal(state[k],v) for k,v in original.items() if k!='projection.weight'):raise ValueError('Functional motion normalization changed')
    if float(abs(state['projection.weight']-original['projection.weight']).max())>cp['maximum_weight_change']+1e-7:raise ValueError('Functional projection exceeds change bound')
    motion.load_state_dict(state,strict=True);motion.eval().requires_grad_(False)
    return dict(receipt,functional_checkpoint_sha256=sha256(path),functional_scope='Synthetic functional research overlay only; not live/actor promotion',actor_qualified=False,vehicle_control_sent=False)
