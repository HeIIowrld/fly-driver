"""Same-camera/native runtime with a bounded visual speed input; no actuators."""
import torch
from .motion_spatial_runtime import MotionSpatialDecisionRuntime
from .visual_flight_speed_intent import VisualFlightSpeedIntent

class TurnSpeedDecisionRuntime(MotionSpatialDecisionRuntime):
 def __init__(self,*components,motion_port,speed_intent,speed_active=True,**kwargs):
  if type(speed_intent) is not VisualFlightSpeedIntent or type(speed_active) is not bool:raise ValueError('Explicit visual speed adapter and mode required')
  if any(t.device.type!='cpu' for t in [*speed_intent.parameters(),*speed_intent.buffers()]):raise ValueError('Fixed CPU speed adapter required')
  self.speed_intent=speed_intent.eval().requires_grad_(False);self.speed_active=speed_active;self.last_speed_trace=None
  super().__init__(*components,motion_port=motion_port,**kwargs);self.handles.append(self.model.motor.command.register_forward_pre_hook(self._speed_input))
 def _speed_input(self,module,args):
  if self.active is None:raise ValueError('Original admitted native/current camera context unavailable')
  command,body,valid,goals=args[:4]
  if not torch.equal(body,self.active['body']):raise ValueError('Speed and native context belong to different bodies')
  target,correction,gain=self.speed_intent.evaluate(self.active['static'],body[:,16:17],valid[:,16:17],self.active['native'],self.active['valid'],torch.ones(1,4,dtype=torch.bool),goals)
  self.last_speed_trace=dict(desired_speed_m_s=float(target[0,0]),speed_correction_m_s=float(correction[0,0]),native_gain=float(gain[0,0]),speed_intent_active=self.speed_active)
  if not self.speed_active:return args
  return (torch.cat([command[:,:1],target],1),*args[1:])
 def decide(self,**kwargs):
  started=self.monotonic();result=super().decide(**kwargs);elapsed=self.monotonic()-started;age=kwargs['decision_time']-kwargs['body_context'].decision_body.original_received_monotonic_seconds
  if elapsed<0 or age+elapsed>.05:self.poisoned=True;raise ValueError('Complete speed/binding inference exceeds body50ms contract')
  result.update(self.last_speed_trace or {});result.update(inference_seconds=elapsed,estimated_output_body_age_seconds=age+elapsed);return result
