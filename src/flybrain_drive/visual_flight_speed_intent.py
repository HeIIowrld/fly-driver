"""Engineering retinal speed intent with bounded native-body modulation.

No measured fly tuning, friction lookup, route, world pose or vehicle actuator.
A zero retinal projection yields zero speed correction regardless of body state.
The current body speed is subtracted once by the existing ManeuverCurrents.
"""
import math
import torch
from torch import nn

class VisualFlightSpeedIntent(nn.Module):
    def __init__(self,reference,scale,native_reference,native_scale,*,max_delta_m_s=10.,maximum_log_gain=.5):
        super().__init__()
        for mean,std,size in ((reference,scale,320),(native_reference,native_scale,19)):
            if mean.shape!=(size,) or std.shape!=mean.shape or mean.dtype!=torch.float32 or std.dtype!=mean.dtype or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():raise ValueError('Original finite TRAIN normalizations required')
        if not 0<max_delta_m_s<=20 or not 0<maximum_log_gain<=.5 or not all(math.isfinite(v) for v in (max_delta_m_s,maximum_log_gain)):raise ValueError('Explicit bounded speed and positive body gain required')
        for name,value in [('reference',reference),('scale',scale),('native_reference',native_reference),('native_scale',native_scale)]:self.register_buffer(name,value.detach().clone())
        self.max_delta_m_s=float(max_delta_m_s);self.maximum_log_gain=float(maximum_log_gain)
        self.visual_projection=nn.Linear(320,1,bias=False);self.body_gain=nn.Linear(38,1,bias=False)
        nn.init.zeros_(self.visual_projection.weight);nn.init.zeros_(self.body_gain.weight)

    def evaluate(self,features,speed,speed_valid,native,native_valid,view_valid,goals):
        tensors=((features,(1,320)),(speed,(1,1)),(native,(1,19)),(goals,(1,2)))
        for value,shape in tensors:
            if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=self.reference.dtype or value.device!=self.reference.device or not torch.isfinite(value).all():raise ValueError('Finite current retinal/native/speed/goal tensors required')
        for mask,shape in ((speed_valid,(1,1)),(native_valid,(1,19)),(view_valid,(1,4))):
            if not isinstance(mask,torch.Tensor) or mask.shape!=shape or mask.dtype!=torch.bool or mask.device!=features.device:raise ValueError('Explicit availability masks required')
        if not speed_valid.all() or not view_valid.all() or (speed<0).any() or (goals[:,:1]!=0).any() or (goals[:,1:]<=0).any() or (goals[:,1:]>3.5).any():raise ValueError('Valid four-view camera and unsigned body speed/racing ceiling required')
        retinal=((features-self.reference)/self.scale).clamp(-5,5)
        z=((native-self.native_reference)/self.native_scale).clamp(-5,5);z=torch.where(native_valid,z,torch.zeros_like(z))
        encoded=torch.cat([z,native_valid.to(features.dtype)-1],dim=1)
        log_gain=self.maximum_log_gain*torch.tanh(self.body_gain(encoded))
        gain=torch.where(native_valid.any(1,keepdim=True),torch.exp(log_gain),torch.ones_like(log_gain))
        retinal_intent=self.visual_projection(retinal)
        correction=self.max_delta_m_s*torch.tanh(retinal_intent*gain)
        target=torch.minimum((speed+correction).clamp_min(0),goals[:,1:]*100/3.6)
        return target,correction,gain

    def forward(self,*args):return self.evaluate(*args)[0]
