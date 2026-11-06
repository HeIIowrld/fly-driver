"""Freeze engineering speed coefficients while preserving the visual input adjoint."""
import torch
from torch.nn import functional as F
from .symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand

def frozen_parameter_speed_command(module,muscles,speed,racing_goals):
    """Same numeric mapping, no coefficient gradient, intact muscle-input gradient."""
    if type(module) is not SymmetricNormalizedFlightSpeedCommand:
        raise ValueError('Exact symmetric normalized speed boundary required')
    weight=module.projection.weight
    if (not isinstance(racing_goals,torch.Tensor) or racing_goals.shape!=(1,2)
            or racing_goals.dtype!=weight.dtype or racing_goals.device!=weight.device
            or not torch.isfinite(racing_goals).all() or (racing_goals[:,0]!=0).any()
            or ((racing_goals[:,1]<=0)|(racing_goals[:,1]>3.5)).any()):
        raise ValueError('Original finite racing ceiling required')
    features=module.encode(muscles,speed)
    correction=module.max_delta_m_s*torch.tanh(F.linear(features,weight.detach(),module.projection.bias.detach()))
    target=torch.minimum((speed+correction).clamp_min(0.),racing_goals[:,1:2]*(100./3.6))
    return torch.cat([torch.zeros_like(target),target],dim=1)
