"""Strict TRAIN-normalized symmetric speed overlay, read-only research runtime."""
from copy import deepcopy
from pathlib import Path
import torch
from .graph import sha256
from .sac_unified_student import read
from .flight_speed_policy_candidate import apply_flight_speed_policy,FlightSpeedPolicyDecisionRuntime
from .spatial_motion_dn_candidate import SpatialMotionDNDecisionRuntime
from .symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand

def apply_symmetric_speed(checkpoint,components,calibration,motion):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];p=m['plan'];normpath=path.parent.parent/'normalization-01.json'
 if r['status']!='completed_symmetric_speed_fit' or sha256(path)!=r['checkpoint_sha256'] or any(sha256(p[k])!=p[k+'_sha256'] for k in ['normalized_parent_checkpoint','normalized_parent_optimizer','motion_parent_checkpoint','fixed_decoder']) or sha256(normpath)!=p['normalization_sha256'] or read(normpath)!=p['speed_normalization']:raise ValueError('Completed hash-bound TRAIN-normalized speed inputs required')
 cp=torch.load(path,map_location='cpu',weights_only=True);keys={'schema','arm','current_calibration','body_port','fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','motion_parent_checkpoint_sha256','speed_parent_checkpoint_sha256','speed_parent_optimizer_sha256','normalized_parent_checkpoint_sha256','normalization_sha256','speed_policy','speed_max_delta_m_s','initial_gate','minimum_gate','actor_qualified'}
 if set(cp)!=keys or cp['schema']!='flybrain.symmetric_normalized_speed_learning.experimental.v1' or cp['arm'] not in ['neutral_control','speed_policy'] or cp['arm']!=m['arm'] or cp['actor_qualified'] is not False or any(cp[k]!=p[k] for k in keys-{'schema','arm','current_calibration','body_port','speed_policy','actor_qualified'}):raise ValueError('Exact experimental symmetric speed schema required')
 if components[0].speed_command_boundary is not None:raise ValueError('Fresh source model required')
 # Parent loader attaches a child: copy the model as well as native port before validation.
 private=(deepcopy(components[0]),*components[1:4],deepcopy(components[4]),*components[5:]);changed,current,receipt=apply_flight_speed_policy(p['normalized_parent_checkpoint'],private,deepcopy(calibration),motion);body=changed[4];n=p['speed_normalization'];policy=SymmetricNormalizedFlightSpeedCommand(torch.tensor(n['reference'],dtype=torch.float32),torch.tensor(n['scale'],dtype=torch.float32),max_delta_m_s=p['speed_max_delta_m_s'])
 for module,key,allowed in [(current,'current_calibration',{'motion_projection.weight'}),(body,'body_port',{'projection.weight'}),(policy,'speed_policy',{'projection.weight','projection.bias'})]:
  expected=module.state_dict();state=cp[key]
  if set(state)!=set(expected) or any(not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k not in allowed and not torch.equal(v,expected[k]) for k,v in state.items()):raise ValueError('Exact frozen TRAIN normalization and finite parameters required')
 old=torch.load(p['motion_parent_checkpoint'],map_location='cpu',weights_only=True)['body_port']['projection.weight'];w=cp['current_calibration']['motion_projection.weight'];delta=cp['body_port']['projection.weight']-old;sw=cp['speed_policy']['projection.weight'];sb=cp['speed_policy']['projection.bias']
 if w.norm()>p['visual_total_norm']+1e-7 or w.abs().max()>2+1e-7 or delta.norm()>p['native_total_norm']+1e-7 or delta.abs().max()>p['native_entry_delta']+1e-7 or sw.norm()>p['speed_weight_total_norm']+1e-7 or sw.abs().max()>p['speed_weight_absolute_bound']+1e-7 or sb.norm()>p['speed_bias_total_norm']+1e-7 or sb.abs().max()>p['speed_bias_absolute_bound']+1e-7 or sw[0,-1]>0 or cp['arm']=='neutral_control' and (torch.count_nonzero(sw)>0 or torch.count_nonzero(sb)>0):raise ValueError('Cumulative motion/native bounds and projected speed gain required')
 current.load_state_dict(cp['current_calibration']);body.load_state_dict(cp['body_port']);policy.load_state_dict(cp['speed_policy']);current.eval().requires_grad_(False);body.eval().requires_grad_(False);policy.eval().requires_grad_(False);changed[0].speed_command_boundary=policy
 return changed,current,dict(receipt,symmetric_speed_checkpoint_sha256=sha256(path),normalization_sha256=p['normalization_sha256'],arm=cp['arm'],actor_qualified=False,vehicle_control_sent=False)

class SymmetricSpeedDecisionRuntime(FlightSpeedPolicyDecisionRuntime):
 def __init__(self,*components,policy_limits,**kwargs):
  policy=components[0].speed_command_boundary
  if type(policy) is not SymmetricNormalizedFlightSpeedCommand or policy.muscle_count!=6:raise ValueError('Explicit TRAIN-normalized symmetric speed child required')
  self.speed_policy=policy;self.policy_limits=dict(policy_limits);self._check_speed_policy();SpatialMotionDNDecisionRuntime.__init__(self,*components,**kwargs)
 def _check_speed_policy(self):
  super()._check_speed_policy();module=self.speed_policy;n=self.policy_limits['speed_normalization']
  for key in ['reference','scale']:
   v=getattr(module,key);expected=torch.tensor(n[key],dtype=torch.float32)
   if v.device.type!='cpu' or v.dtype!=expected.dtype or v.shape!=(6,) or not torch.equal(v,expected):raise ValueError('Fixed TRAIN muscle normalization changed')
