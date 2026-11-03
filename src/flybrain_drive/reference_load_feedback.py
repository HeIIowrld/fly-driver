"""Experimental reference-centered load port; not a physiological or actor claim."""
import torch
from torch import nn
from .flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter

class ReferenceLoadFeedback(nn.Module):
    """Add old experimental load response relative to one explicit body reference."""
    def __init__(self, base, load_weights, reference):
        super().__init__()
        if type(base) is not CenteredSpeedBodyFeedbackAdapter:
            raise ValueError('Exact existing centered speed feedback required')
        w=torch.as_tensor(load_weights,dtype=base.scales.dtype,device=base.scales.device).detach().clone()
        r=torch.as_tensor(reference,dtype=base.scales.dtype,device=base.scales.device).detach().clone()
        if w.shape!=(base.projection.out_features,4) or r.shape!=(4,) or not torch.isfinite(w).all() or not torch.isfinite(r).all() or torch.any(r<0) or not torch.isclose(r.sum(),r.new_tensor(1.),atol=1e-5,rtol=0):
            raise ValueError('Finite port weights and normalized explicit load reference required')
        self.base=base
        self.register_buffer('load_weights',w)
        self.register_buffer('reference',r)

    def forward(self,body,body_valid):
        # Call base first to keep its original shape, device and finiteness checks.
        original=self.base(body,body_valid)
        if not body_valid[:,8:12].all():
            raise ValueError('All four original load measurements required; no imputation')
        loads=body[:,8:12]
        if torch.any(loads<0) or not torch.allclose(loads.sum(dim=1),torch.ones_like(loads[:,0]),atol=1e-5,rtol=0):
            raise ValueError('Normalized nonnegative load fractions required')
        available=body_valid[:,self.base.indices]
        values=torch.tanh(body[:,self.base.indices]/self.base.scales)
        values=torch.where(available,values,torch.zeros_like(values))
        logits=self.base.projection(torch.cat([values,available.to(body.dtype)-1.],dim=1))
        delta=(torch.tanh(loads)-torch.tanh(self.reference))@self.load_weights.T
        # Exact baseline is guaranteed by evaluating the same base output.
        return original + .5*(torch.tanh(logits+delta)-torch.tanh(logits))
