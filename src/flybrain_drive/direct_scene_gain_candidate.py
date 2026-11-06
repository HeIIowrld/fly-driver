"""Strict read-only restoration of direct visual-port gain continuation candidates."""
from pathlib import Path
import torch
from .current_flight_runtime import load_current_flight_runtime
from .direct_scene_port_gain import DirectScenePortGain
from .graph import sha256
from .sac_unified_student import read
from .power_feedback_sensitivity import power_sensitivity


def validate_direct_gain_state(state,reference,arm):
 expected={'reference':reference,'delta':torch.zeros_like(reference)}
 if arm not in ['frozen_direct','learn_direct'] or set(state)!=set(expected) or any(not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Exact finite direct visual gain schema required')
 if not torch.equal(state['reference'],reference) or state['delta'].norm()>.5+1e-7 or state['delta'].abs().max()>.25+1e-7 or arm=='frozen_direct' and torch.count_nonzero(state['delta']):raise ValueError('TRAIN reference and bounded direct visual gain required')


def load_direct_scene_gain_runtime(checkpoint,expected_sha256):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];p=m['plan'];runtime=None
 if r['status']!='completed_direct_scene_gain_fit' or sha256(path)!=expected_sha256 or expected_sha256!=r['checkpoint_sha256'] or any(sha256(p[k])!=p[k+'_sha256'] for k in ['direct_parent_checkpoint','direct_parent_optimizer','motion_parent_checkpoint','fixed_decoder']) or any(sha256(f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Completed exact direct visual gain continuation required')
 refpath=path.parent.parent/'direct-reference-01.json'
 if sha256(refpath)!=p['direct_gain_reference_sha256'] or read(refpath)['reference']!=p['direct_gain_reference'] or read(refpath)['development_used'] is not False:raise ValueError('Pinned TRAIN-only direct reference required')
 cp=torch.load(path,map_location='cpu',weights_only=True);parent=torch.load(p['direct_parent_checkpoint'],map_location='cpu',weights_only=True);extras={'direct_gain','direct_parent_checkpoint_sha256','direct_parent_optimizer_sha256','direct_gain_reference_sha256'}
 if set(cp)!=set(parent)|extras or cp['schema']!='flybrain.direct_scene_gain_learning.experimental.v1' or cp['arm']!=m['arm'] or cp['actor_qualified'] is not False or cp['fixture_input_adjoint'] is not True or any(cp[k]!=p[k] for k in extras-{'direct_gain'}):raise ValueError('Exact experimental candidate schema required')
 variable={'schema','arm','current_calibration','body_port','speed_policy'}
 if any(cp[k]!=parent[k] for k in set(parent)-variable):raise ValueError('Parent provenance metadata changed')
 for key,allowed in [('current_calibration',{'motion_projection.weight'}),('body_port',{'projection.weight'}),('speed_policy',{'projection.weight','projection.bias'})]:
  state,expected=cp[key],parent[key]
  if set(state)!=set(expected) or any(not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k not in allowed and not torch.equal(v,expected[k]) for k,v in state.items()):raise ValueError('Frozen graph/input normalization changed')
 reference=torch.tensor(p['direct_gain_reference'],dtype=torch.float32);validate_direct_gain_state(cp['direct_gain'],reference,cp['arm']);old=torch.load(p['motion_parent_checkpoint'],map_location='cpu',weights_only=True)['body_port']['projection.weight'];w=cp['current_calibration']['motion_projection.weight'];delta=cp['body_port']['projection.weight']-old;sw=cp['speed_policy']['projection.weight'];sb=cp['speed_policy']['projection.bias']
 if w.norm()>p['visual_total_norm']+1e-7 or w.abs().max()>2+1e-7 or delta.norm()>p['native_total_norm']+1e-7 or delta.abs().max()>p['native_entry_delta']+1e-7 or sw.norm()>p['speed_weight_total_norm']+1e-7 or sw.abs().max()>p['speed_weight_absolute_bound']+1e-7 or sb.norm()>p['speed_bias_total_norm']+1e-7 or sb.abs().max()>p['speed_bias_absolute_bound']+1e-7 or sw[0,-1]>0 or float(power_sensitivity(sw,cp['speed_policy']['scale'],p['speed_max_delta_m_s']))>p['maximum_power_input_sensitivity']+1e-3:raise ValueError('Original cumulative motion/native/speed bounds failed')
 try:
  runtime,receipt=load_current_flight_runtime(p['direct_parent_checkpoint'],p['direct_parent_checkpoint_sha256']);runtime.current_calibration.load_state_dict(cp['current_calibration']);runtime.body_port.load_state_dict(cp['body_port']);runtime.speed_policy.load_state_dict(cp['speed_policy']);runtime.policy_limits=p;runtime._check_speed_policy();gain=DirectScenePortGain(reference);gain.load_state_dict(cp['direct_gain']);gain.eval().requires_grad_(False);runtime.direct_gain=gain;runtime.handles.append(runtime.model.visual.register_forward_hook(lambda module,args,value:gain(value),prepend=True))
  return runtime,dict(parent_receipt=receipt,checkpoint_sha256=expected_sha256,arm=cp['arm'],direct_gain_reference_sha256=p['direct_gain_reference_sha256'],direct_gain_change_norm=float(gain.delta.norm()),actor_qualified=False,vehicle_control_sent=False)
 except Exception:
  if runtime is not None:runtime.close()
  raise
