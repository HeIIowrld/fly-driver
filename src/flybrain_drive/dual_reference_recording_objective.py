"""Training-only protection against both original and immediate-parent errors.

References are loss targets, never actor observations. This is a soft objective,
not an acceptance gate or a closed-loop driving guarantee.
"""
import torch
from .functional_priority_objective import recording_losses


def dual_reference_recording_losses(prediction,anchor,parent_anchor,target,fit_mask,pre_mask,*,anchor_weight,maximum_ratio,protection_weight):
 supervision,preservation,_=recording_losses(prediction,anchor,target,fit_mask,pre_mask,anchor_weight=anchor_weight,maximum_ratio=maximum_ratio,protection_weight=protection_weight)
 if not isinstance(parent_anchor,torch.Tensor) or parent_anchor.shape!=anchor.shape or parent_anchor.dtype!=anchor.dtype or parent_anchor.device!=anchor.device or not torch.isfinite(parent_anchor).all():raise ValueError('Aligned finite immediate-parent loss reference required')
 violations=[]
 for mask,axes in ((fit_mask,[0,1,2]),(pre_mask&fit_mask,[0])):
  if not mask.any():continue
  now=(prediction[mask][:,axes]-target.detach()[mask][:,axes]).square().mean(0);reference_violations=[]
  for reference in [anchor,parent_anchor]:
   old=(reference.detach()[mask][:,axes]-target.detach()[mask][:,axes]).square().mean(0);reference_violations.append(torch.relu((now-maximum_ratio**2*old)/old.clamp_min(1e-5)))
  violations.append(torch.maximum(*reference_violations))
 protection=prediction.sum()*0 if not violations else protection_weight*torch.cat(violations).square().mean()
 return supervision,preservation,protection
