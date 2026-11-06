"""Bound an engineering speed boundary's direct power-MN input sensitivity."""
import torch

def power_sensitivity(weight,scale,max_delta_m_s):
    return max_delta_m_s*(weight[0,2].abs()/scale[2]+weight[0,5].abs()/(2*scale[5]))

@torch.no_grad()
def project_power_sensitivity(weight,scale,*,max_delta_m_s,maximum):
    """Euclidean projection of two coefficients onto a weighted L1 ball.

    This bounds each power-MN input partial at the speed command. It is not a
    bound on the complete circuit Jacobian or a physical stability certificate.
    """
    if (weight.shape!=(1,7) or scale.shape!=(6,) or weight.dtype!=torch.float32
            or scale.dtype!=torch.float32 or weight.device.type!='cpu'
            or scale.device.type!='cpu' or not torch.isfinite(weight).all()
            or not torch.isfinite(scale).all() or (scale<=0).any()
            or not 0<float(maximum)<=100000 or not 0<float(max_delta_m_s)<=20):
        raise ValueError('Explicit finite CPU speed coefficients and positive sensitivity bound required')
    indices=[2,5];v=weight[0,indices].double();a=max_delta_m_s*torch.stack([1/scale[2].double(),1/(2*scale[5].double())]);absolute=v.abs()
    if float((a*absolute).sum())<=maximum:return
    low=0.;high=float((absolute/a).max())
    for _ in range(64):
        mid=(low+high)/2
        if float((a*(absolute-mid*a).clamp_min(0)).sum())>maximum:low=mid
        else:high=mid
    projected=(v.sign()*(absolute-high*a).clamp_min(0)).float();weight[0,indices]=projected
    # F32 rounding can put the weighted sum just outside the mathematical ball.
    sensitivity=float(power_sensitivity(weight,scale,max_delta_m_s))
    if sensitivity>maximum:weight[0,indices]*=maximum/sensitivity
    if float(power_sensitivity(weight,scale,max_delta_m_s))>maximum+1e-3:raise ValueError('Power sensitivity projection failed')
