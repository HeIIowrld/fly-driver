"""Experimental all-or-none load feedback; not a vehicle control qualification."""
import torch
from torch import nn
from .flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter


class CompleteWheelLoadFeedback(nn.Module):
    """Use the candidate's load values only when all four original loads are valid.

    Availability indicators remain unchanged. No invalid load is substituted with
    a fabricated measurement. This experimental wrapper has no actor admission.
    """
    def __init__(self, base):
        super().__init__()
        if type(base) is not CenteredSpeedBodyFeedbackAdapter:
            raise ValueError('Exact centered speed body adapter required')
        self.base=base
        if self.base.indices.tolist()!=[0,1,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            raise ValueError('Expected original body feature ordering')

    def forward(self,body,body_valid):
        # Preserve every validation of the original implementation.
        self.base(body,body_valid)
        availability=body_valid[:,self.base.indices]
        raw=torch.tanh(body[:,self.base.indices]/self.base.scales)
        feature_mask=availability.clone()
        feature_mask[:,2:6] &= body_valid[:,8:12].all(dim=1,keepdim=True)
        values=torch.where(feature_mask,raw,torch.zeros_like(raw))
        encoded=torch.cat((values,availability.to(body.dtype)-1.),dim=1)
        return .5*torch.tanh(self.base.projection(encoded))
