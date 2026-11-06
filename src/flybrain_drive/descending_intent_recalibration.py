"""Strict offline-only overlay for existing 320->1 visual DN intent calibration."""
from pathlib import Path
import torch
from .functional_motion_overlay import exact_model
from .graph import sha256
from .sac_unified_student import read

def load_descending_intent(path,components,motion,receipt):
    path=Path(path);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];cp=torch.load(path,map_location='cpu',weights_only=True)
    if (r['status']!='completed_descending_intent_fit' or cp['schema']!='flybrain.descending_intent.v1' or sha256(path)!=r['checkpoint_sha256'] or cp['actor_qualified'] is not False or cp['parent_candidate']!=receipt or m['parent_candidate']!=receipt or cp['optimizer_updates']!=r['optimizer_updates']):raise ValueError('Unqualified hash-bound intent calibration required')
    for key,index in [('model',0),('readout',1),('adapter',2),('optical',3),('body_port',4),('motion_port',None)]:
        current=(motion if index is None else components[index]).state_dict();given=cp[key]
        if key=='adapter':
            if set(current)!=set(given):raise ValueError('Adapter keys changed')
            weight=given['projection.weight'];original=current['projection.weight']
            if weight.shape!=original.shape or weight.dtype!=original.dtype or not torch.isfinite(weight).all() or float(abs(weight-original).max())>m['plan']['maximum_weight_delta']+1e-7:raise ValueError('Intent projection exceeds original contract')
            current={k:v for k,v in current.items() if k!='projection.weight'};given={k:v for k,v in given.items() if k!='projection.weight'}
        if not exact_model(current,given):raise ValueError('Fixed component altered '+key)
    components[2].load_state_dict(cp['adapter'],strict=True);components[2].eval().requires_grad_(False)
    return dict(receipt,descending_intent_checkpoint_sha256=sha256(path),descending_intent_scope='Existing engineering DN input calibration only; no live or actuator authority',actor_qualified=False,vehicle_control_sent=False)
