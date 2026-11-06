"""Existing recurrent flight state commands physical speed error through DNg02."""
from copy import deepcopy
from dataclasses import replace
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.functional_motion_overlay import exact_model
from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from scripts.spatial_motion_dn_experiment import SpatialMotionDNExperiment
from scripts.visual_native_port_experiment import bounded_step
from scripts.train_qualified_sac_unified import context

class FlightSpeedPolicyExperiment(SpatialMotionDNExperiment):
 def __init__(self,plan,*,learn_speed):
  super().__init__(plan,train_body=True);self.learn_speed=bool(learn_speed)
  if sha256(plan['speed_parent_checkpoint'])!=plan['speed_parent_checkpoint_sha256']:raise ValueError('Exact trained motion-DN source required')
  cp=torch.load(plan['speed_parent_checkpoint'],map_location='cpu',weights_only=True)
  if cp['schema']!='flybrain.spatial_motion_dn_learning.experimental.v1' or cp['actor_qualified'] is not False:raise ValueError('Explicit experimental motion-DN source required')
  self.calibration.load_state_dict(cp['current_calibration']);self.body_port.load_state_dict(cp['body_port']);self.baseline_calibration=deepcopy(self.calibration).eval().requires_grad_(False);self.baseline_native_port=deepcopy(self.body_port).eval().requires_grad_(False)
  if self.model.speed_command_boundary is not None:raise ValueError('Source speed path must be unconnected')
  self.speed_policy=FlightStateSpeedCommand(6,max_delta_m_s=plan['speed_max_delta_m_s']);self.speed_policy.requires_grad_(self.learn_speed);self.model.speed_command_boundary=self.speed_policy;self.speed_origin={k:v.clone() for k,v in self.speed_policy.state_dict().items()};self.last_command=None;self.last_power=None;self.last_body_speed=None;self.hooks.extend([self.speed_policy.register_forward_hook(self._speed_policy_hook),self.model.motor.command.register_forward_pre_hook(self._fixed_reference),self.model.motor.command.register_forward_hook(self._record_maneuver)])
 def _speed_policy_hook(self,module,args,value):
  if self.holder.get('speed_policy_off',False):
   speed=args[1];target=torch.minimum(speed,args[2][:,1:2]*(100./3.6));return torch.cat([torch.zeros_like(target),target],1)
  if self.holder.get('mode')=='synthetic' or self.holder.get('counterfactual_images',False):return value.detach()
  return value
 def _fixed_reference(self,module,args):
  h=self.holder
  if 'fixed_command_trace' not in h:return args
  command=args[0].clone();command[0]=torch.as_tensor(h['fixed_command_trace'][h['index']-h['case_begin']],dtype=command.dtype,device=command.device);return (command,*args[1:])
 def _record_maneuver(self,module,args,value):
  self.last_command=args[0].detach().clone();self.last_body_speed=args[1][:,16:17].detach().clone();self.last_power=value[1].detach().clone()
 def train_parameters(self):return [self.calibration.projection.weight,self.body_port.projection.weight]+([self.speed_policy.projection.weight,self.speed_policy.projection.bias] if self.learn_speed else [])
 def learning_rates(self):
  p=self.port_plan;return [p['visual_learning_rate'],p['native_learning_rate']]+([p['speed_weight_learning_rate'],p['speed_bias_learning_rate']] if self.learn_speed else [])
 def make_optimizer(self):
  p=self.port_plan;path=p['speed_parent_optimizer'];assert sha256(path)==p['speed_parent_optimizer_sha256'];parameters=self.train_parameters();optimizer=torch.optim.Adam([dict(params=[w],lr=lr) for w,lr in zip(parameters[:2],self.learning_rates()[:2])]);optimizer.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));assert [int(optimizer.state[w]['step']) for w in parameters[:2]]==p['initial_optimizer_steps'];assert [g['lr'] for g in optimizer.param_groups]==self.learning_rates()[:2]
  for w,lr in zip(parameters[2:],self.learning_rates()[2:]):optimizer.add_param_group(dict(params=[w],lr=lr))
  return optimizer
 def project_after_step(self,old):
  super().project_after_step(old[:2]);p=self.port_plan
  if self.learn_speed:
   w,b=self.speed_policy.projection.weight,self.speed_policy.projection.bias
   if w.grad is not None:
    bounded_step(w,old[2],torch.zeros_like(w),step_norm=p['speed_weight_step_norm'],total_norm=p['speed_weight_total_norm'],absolute=p['speed_weight_absolute_bound'])
    with torch.no_grad():w[0,-1].clamp_(max=0.)
   if b.grad is not None:bounded_step(b,old[3],torch.zeros_like(b),step_norm=p['speed_bias_step_norm'],total_norm=p['speed_bias_total_norm'],absolute=p['speed_bias_absolute_bound'])
 def _body_context(self,ctx):
  h=self.holder;dy=float(h.get('sensed_yaw_delta',0.));dv=float(h.get('sensed_speed_delta',0.))
  if dy==dv==0:return ctx
  def shifted(obs):
   if obs is None:return None
   assert obs.valid[[4,16]].all();v=obs.values.copy();v[4]+=dy;v[16]+=dv;assert v[16]>=0;return replace(obs,values=v)
  return per_view_temporal_body_context(tuple(shifted(v) for v in ctx.visual_contexts),shifted(ctx.decision_body),render_start_lowers=ctx.render_start_lowers,camera_receipt_time=ctx.camera_receipt_time,decision_time=ctx.decision_time)
 def step(self,i,learn):
  h=self.holder;h['index']=i
  if h['mode']=='real':
   episode=h['episode'];d,c,_=self.items[episode];donor=h.get('donor');de,di=donor if donor is not None else (episode,i);scene=torch.tensor(self.items[de][0]['scene'][di:di+1]);ctx=self._body_context(context(d,c,i));now=float(d['decision_time'][i])
  else:
   x=h['source']['x'];clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;values,mask=h['body'],h['mask'];current=NavigationBodyObservation('synthetic-factorial',1000+i,now-.001,now,values.copy(),mask.copy(),self.contract);history=tuple(NavigationBodyObservation('synthetic-factorial',10*i+j,t-.001,camera,values.copy(),mask.copy(),self.contract) for j,t in enumerate(clocks));ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now);scene=torch.tensor(x['scene'][h['case'],i:i+1])
  with torch.set_grad_enabled(learn):
   fn=self.model.experimental_recurrent_speed_training_decision if learn else self.model.experimental_recurrent_speed_decision;_,mu=fn(decision_time=now,racing_goals=self.goals,scene_features=scene,scene_valid=self.views,body_context=ctx);return self.head(mu)[0]
 def verify_fixed(self):
  for index,(module,prior) in enumerate(zip([*self.components[:5],self.motion],self.protected)):
   state=module.state_dict()
   if index==0:
    allowed={'speed_command_boundary.projection.weight','speed_command_boundary.projection.bias'};assert allowed.issubset(state);state={k:v for k,v in state.items() if k not in allowed}
    if not exact_model(state,prior):raise ValueError('Original flight graph/legacy sensory components changed')
   elif index==1:
    if not exact_model(state,self.fixed_head):raise ValueError('Fixed signed decoder changed')
   elif index==4:
    if not exact_model({k:v for k,v in state.items() if k!='projection.weight'},self.original_native_buffers):raise ValueError('Native normalization changed')
   elif not exact_model(state,prior):raise ValueError('Frozen sensory component changed')
  state={k:v for k,v in self.calibration.state_dict().items() if k!='motion_projection.weight'};p=self.port_plan;w=self.calibration.projection.weight.detach();native=self.body_port.projection.weight.detach();delta=native-self.native_origin
  if not exact_model(state,self.fixed_overlay) or not torch.equal(self.gain.value,torch.ones_like(self.gain.value)) or not torch.isfinite(w).all() or w.norm()>p['visual_total_norm']+1e-7 or w.abs().max()>2+1e-7 or not torch.isfinite(native).all() or delta.norm()>p['native_total_norm']+1e-7 or delta.abs().max()>p['native_entry_delta']+1e-7:raise ValueError('Frozen parent or cumulative motion/native bounds failed')
  sw,sb=self.speed_policy.projection.weight.detach(),self.speed_policy.projection.bias.detach()
  if self.speed_policy.max_delta_m_s!=p['speed_max_delta_m_s'] or not torch.isfinite(sw).all() or not torch.isfinite(sb).all() or sw.norm()>p['speed_weight_total_norm']+1e-7 or sw.abs().max()>p['speed_weight_absolute_bound']+1e-7 or sb.norm()>p['speed_bias_total_norm']+1e-7 or sb.abs().max()>p['speed_bias_absolute_bound']+1e-7 or sw[0,-1]>0:raise ValueError('Bounded nonpositive direct speed gain required')
  if not self.learn_speed and not exact_model(self.speed_policy.state_dict(),self.speed_origin):raise ValueError('Neutral control policy changed')
