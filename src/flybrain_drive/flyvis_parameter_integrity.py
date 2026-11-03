"""Validate and constrain experimental FlyVis synapse strengths."""
import math
import torch


def assert_nonnegative_synapse_strength(values):
    if not isinstance(values, torch.Tensor) or not torch.is_floating_point(values) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('FlyVis synapse strength must be finite and nonnegative')


def project_nonnegative_bounded(candidate, original, allowed, max_change):
    """Preserve non-target elements and enforce FlyVis nonnegative strengths."""
    assert_nonnegative_synapse_strength(original)
    if (not isinstance(candidate, torch.Tensor) or candidate.shape != original.shape
            or candidate.dtype != original.dtype or candidate.device != original.device
            or not torch.isfinite(candidate).all() or not isinstance(allowed, torch.Tensor)
            or allowed.shape != original.shape or allowed.dtype != torch.bool or allowed.device != original.device
            or isinstance(max_change, bool) or not isinstance(max_change, (int, float))
            or not math.isfinite(max_change) or max_change <= 0):
        raise ValueError('Finite matching parameters, mask and positive bound required')
    lower = torch.clamp(original - max_change, min=0.)
    upper = original + max_change
    result = torch.where(allowed, torch.maximum(torch.minimum(candidate, upper), lower), original)
    assert_nonnegative_synapse_strength(result)
    return result
