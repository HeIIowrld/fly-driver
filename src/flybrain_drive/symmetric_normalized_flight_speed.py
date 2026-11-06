"""TRAIN-normalized symmetric six-muscle command at existing flight speed boundary."""
import torch
from .flight_state_speed_command import FlightStateSpeedCommand

class SymmetricNormalizedFlightSpeedCommand(FlightStateSpeedCommand):
    """Engineering registration: three abs pair differences and three pair means.

    No new measured synapse, driver head, oracle input or recurrent state.
    """
    def __init__(self,reference,scale,*,max_delta_m_s=10.):
        super().__init__(6,max_delta_m_s=max_delta_m_s)
        if (not isinstance(reference,torch.Tensor) or reference.shape!=(6,)
                or not isinstance(scale,torch.Tensor) or scale.shape!=(6,)
                or reference.dtype!=torch.float32 or scale.dtype!=reference.dtype
                or reference.device.type!='cpu' or scale.device!=reference.device
                or not torch.isfinite(reference).all() or not torch.isfinite(scale).all()
                or (scale<1e-5).any()):
            raise ValueError('Explicit finite TRAIN six-feature reference and std floor1e-5 required')
        self.register_buffer('reference',reference.detach().clone())
        self.register_buffer('scale',scale.detach().clone())
    @staticmethod
    def symmetric_features(muscles):
        left,right=muscles[:,::2],muscles[:,1::2]
        return torch.cat([(right-left).abs(),(right+left)/2],dim=1)
    def encode(self,muscles,speed):
        if (muscles.shape!=(1,6) or speed.shape!=(1,1)
                or any(t.device!=self.reference.device or t.dtype!=self.reference.dtype
                       or not torch.isfinite(t).all() for t in [muscles,speed])
                or (speed<0).any() or not torch.isfinite(self.reference).all()
                or not torch.isfinite(self.scale).all() or (self.scale<1e-5).any()):
            raise ValueError('Original finite six-muscle state, body speed and fixed normalization required')
        encoded=((self.symmetric_features(muscles)-self.reference)/self.scale).clamp(-5,5)
        return torch.cat([encoded,speed/30.],dim=1)
    def forward(self,muscles,speed,racing_goals):
        # Reuse the original explicit shape/device/dtype/speed/goal contract.
        super().forward(muscles,speed,racing_goals)
        correction=self.max_delta_m_s*torch.tanh(self.projection(self.encode(muscles,speed)))
        target=torch.minimum((speed+correction).clamp_min(0.),racing_goals[:,1:2]*(100./3.6))
        return torch.cat([torch.zeros_like(target),target],dim=1)
