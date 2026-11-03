"""Differentiable counterpart of the synthetic slip-free bicycle, offline only.

World position remains a loss/evaluation variable, never an actor observation.
This is not a tyre, load transfer, or Assetto Corsa physics model.
"""
import math
import torch


def differentiable_bicycle_step(state, steer, acceleration, *, dt_s, wheelbase_m=2.5):
    """Return (..., x/y/yaw/speed) with exact constant-control integration."""
    if not isinstance(state,torch.Tensor) or state.shape[-1:]!=(4,) or not state.is_floating_point():
        raise ValueError('Floating state tensor (..., 4) required')
    if not isinstance(steer,torch.Tensor) or not isinstance(acceleration,torch.Tensor):
        raise ValueError('Tensor steering and acceleration required')
    if steer.shape!=state.shape[:-1] or acceleration.shape!=state.shape[:-1] or steer.device!=state.device or acceleration.device!=state.device or steer.dtype!=state.dtype or acceleration.dtype!=state.dtype:
        raise ValueError('Matching batch shape, device and dtype required')
    if not isinstance(dt_s,(float,int)) or isinstance(dt_s,bool) or not math.isfinite(dt_s) or not 0<dt_s<=.2 or not isinstance(wheelbase_m,(float,int)) or isinstance(wheelbase_m,bool) or not math.isfinite(wheelbase_m) or wheelbase_m<=0:
        raise ValueError('Explicit bounded timestep and wheelbase required')
    if not torch.isfinite(state).all() or not torch.isfinite(steer).all() or not torch.isfinite(acceleration).all() or torch.any(state[...,3]<0) or torch.any(steer.abs()>.6) or torch.any((acceleration < -12)|(acceleration > 6)):
        raise ValueError('Finite bounded synthetic state and controls required')
    speed=state[...,3];negative=acceleration<0;safe_deceleration=torch.where(negative,-acceleration,torch.ones_like(acceleration));duration=torch.where(negative,torch.minimum(torch.full_like(speed,float(dt_s)),speed/safe_deceleration),torch.full_like(speed,float(dt_s)))
    distance=(speed*duration+.5*acceleration*duration.square()).clamp_min(0.);angle=torch.tan(steer)/float(wheelbase_m)*distance;half=angle*.5;length=distance*torch.sinc(half/math.pi);heading=state[...,2]+half
    return torch.stack((state[...,0]+length*torch.cos(heading),state[...,1]+length*torch.sin(heading),state[...,2]+angle,(speed+acceleration*float(dt_s)).clamp_min(0.)),dim=-1)
