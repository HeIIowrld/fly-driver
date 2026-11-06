"""Strict unqualified turn/speed candidate loader for offline evaluation only."""
from pathlib import Path
import torch
from .functional_motion_overlay import exact_model
from .visual_flight_speed_intent import VisualFlightSpeedIntent
from .graph import sha256
from .sac_unified_student import read

def load_turn_speed(path,components,motion,receipt):
 path=Path(path);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];plan=m['plan'];cp=torch.load(path,map_location='cpu',weights_only=True)
 if r['status']!='completed_turn_speed_intent_fit' or cp['schema']!='flybrain.turn_speed_intent.v1' or sha256(path)!=r['checkpoint_sha256'] or cp['actor_qualified'] is not False or cp['parent_candidate']!=receipt or m['parent_candidate']!=receipt or cp['optimizer_updates']!=r['optimizer_updates'] or r['arm'] not in ('turn_only','turn_speed') or cp['speed_active'] is not (r['arm']=='turn_speed'):raise ValueError('Hash-bound unqualified turn/speed candidate required')
 for key,index in [('model',0),('readout',1),('adapter',2),('optical',3),('body_port',4),('motion_port',None)]:
  current=(components[index] if index is not None else motion).state_dict();given=cp[key]
  if key=='adapter':
   if set(current)!=set(given):raise ValueError('Adapter keys changed')
   old=current['projection.weight'];weight=given['projection.weight']
   if old.shape!=weight.shape or old.dtype!=weight.dtype or not torch.isfinite(weight).all() or float(abs(weight-old).max())>plan['maximum_weight_delta']+1e-7:raise ValueError('Turn projection exceeds original bound')
   current={k:v for k,v in current.items() if k!='projection.weight'};given={k:v for k,v in given.items() if k!='projection.weight'}
  if not exact_model(current,given):raise ValueError('Protected original component changed '+key)
 spec=cp['speed_limits'];expected=dict(max_delta_m_s=plan['maximum_speed_delta_m_s'],maximum_log_gain=plan['maximum_body_log_gain'])
 if spec!=expected:raise ValueError('Speed/gain limits differ')
 speed=VisualFlightSpeedIntent(components[2].reference,components[2].scale,components[4].reference,components[4].scale,**spec);state=cp['speed_intent'];original=speed.state_dict()
 if set(state)!=set(original):raise ValueError('Speed adapter keys differ')
 for key,value in state.items():
  if value.shape!=original[key].shape or value.dtype!=original[key].dtype or not torch.isfinite(value).all():raise ValueError('Speed tensor contract differs')
  if key.endswith('weight'):
   if float(value.abs().max())>plan['maximum_speed_weight_delta']+1e-7 or not cp['speed_active'] and torch.count_nonzero(value):raise ValueError('Speed weight bound/disabled path differs')
  elif not torch.equal(value,original[key]):raise ValueError('Original speed normalizations changed')
 components[2].load_state_dict(cp['adapter'],strict=True);components[2].eval().requires_grad_(False);speed.load_state_dict(state,strict=True);speed.eval().requires_grad_(False)
 return dict(receipt,turn_speed_checkpoint_sha256=sha256(path),turn_speed_scope='Engineering turn/vision-body speed intents through unchanged flight graph; offline only, no actuator authority',actor_qualified=False,vehicle_control_sent=False),speed,cp['speed_active']
