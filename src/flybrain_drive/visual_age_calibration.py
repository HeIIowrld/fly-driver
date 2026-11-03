"""Training-only confidence decay calibration; never changes hard sensor freshness."""

import math

import torch


@torch.no_grad()
def calibrate_visual_age_decay(model, episodes, scale_seconds):
    """Preserve each view's linear drive at its valid training mean source age.

    Frozen policy anchors must be captured BEFORE this transformation. Only local
    visual input weights change; body ports, masks and recurrent state do not.
    Source ages are retained exactly. This is an engineering confidence experiment,
    not exposure synchronization, optical flow or evidence that a frame is fresher.
    """
    hard_age = model.sensory_calibration['max_view_age']
    if (isinstance(scale_seconds, bool) or not isinstance(scale_seconds, (float, int))
            or not math.isfinite(scale_seconds) or not hard_age <= scale_seconds <= 1.):
        raise ValueError('Confidence scale must be finite, between hard freshness and 1s')
    local = model.adapter.visual_adapter
    old_scale = local.age_scale
    totals = local.weight.new_zeros(local.shape[0], dtype=torch.float64)
    counts = torch.zeros_like(totals)
    for episode in episodes:
        observation = episode['observation']
        ages, valid = observation['view_age'], observation['view_valid']
        if (ages.ndim != 2 or ages.shape[1] != local.shape[0]
                or valid.shape != (*ages.shape, local.shape[1])
                or not torch.isfinite(ages).all() or (ages < 0).any()
                or not torch.isin(valid, valid.new_tensor([0, 1])).all()):
            raise ValueError('Finite original ages and binary per-source validity required')
        observed = valid.bool().any(-1)
        if ((ages > hard_age) & observed).any():
            raise ValueError('Valid stale sources cannot calibrate confidence')
        totals += (ages.double()*observed).sum(0).to(totals.device)
        counts += observed.sum(0).to(counts.device)
    if (counts == 0).any():
        raise ValueError('Each source requires valid training observations')
    reference = (totals/counts).to(local.weight.dtype)
    factors = torch.exp(reference*(1./scale_seconds-1./old_scale))
    if not torch.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('Finite positive confidence compensation required')
    local.weight.mul_(factors[local.port_sources, None])
    local.age_scale = float(scale_seconds)
    return dict(old_scale_seconds=float(old_scale), scale_seconds=float(scale_seconds),
        hard_freshness_seconds=hard_age, valid_observations_per_view=counts.cpu().tolist(),
        training_mean_ages_seconds=reference.cpu().tolist(),
        weight_compensation_per_view=factors.cpu().tolist(),
        input_ages_or_masks_modified=False, fixed_neural_connections_modified=False,
        interpretation='same source-local drive at each view training mean age; '
                       'not equal recurrent trajectories at varying source ages')
