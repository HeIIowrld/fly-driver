"""Optional causal temporal adapter on saved four-view motion features.

The filters and learned projection are engineering hypotheses. They do not claim
to be measured fly neurons, replace FlyVis dynamics, or actuate a vehicle.
"""

import math

import torch
from torch import nn

from .flight_embodied_navigation import modulate_current


class TemporalMotionBoundary(nn.Module):
    """Fast-minus-slow motion traces and a neutral-initialized HS/VS projection."""

    def __init__(self, train_mean, train_scale, port_count, *, fast_seconds=.05,
                 slow_seconds=.2):
        super().__init__()
        if (not isinstance(train_mean, torch.Tensor) or train_mean.shape != (576,)
                or train_scale.shape != train_mean.shape or train_scale.dtype != train_mean.dtype
                or train_scale.device != train_mean.device or not torch.isfinite(train_mean).all()
                or not torch.isfinite(train_scale).all() or torch.any(train_scale <= 0)
                or type(port_count) is not int or port_count < 1
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) for v in (fast_seconds, slow_seconds))
                or not 0 < fast_seconds < slow_seconds):
            raise ValueError('Finite TRAIN normalization, ports and ordered time scales required')
        mask = torch.zeros((4, 9, 16), dtype=torch.bool, device=train_mean.device)
        mask[:, 1:] = True
        self.register_buffer('motion_mask', mask.flatten())
        self._validated_mask_key = None
        self.register_buffer('mean', train_mean.detach().clone())
        self.register_buffer('scale', train_scale.detach().clone())
        self.fast_seconds, self.slow_seconds = float(fast_seconds), float(slow_seconds)
        self.projection = nn.Linear(512, port_count, bias=False,
                                    device=train_mean.device, dtype=train_mean.dtype)
        nn.init.zeros_(self.projection.weight)

    def step(self, scene, view_valid, state, elapsed):
        if (not isinstance(scene, torch.Tensor) or scene.ndim != 2 or scene.shape[1] != 576
                or len(scene) < 1 or scene.dtype != self.mean.dtype
                or scene.device != self.mean.device
                or not isinstance(view_valid, torch.Tensor) or view_valid.shape != (len(scene), 4)
                or view_valid.dtype != torch.bool or view_valid.device != scene.device):
            raise ValueError('All four valid ordered views and finite scene features required')
        if not (torch.isfinite(scene).all() & view_valid.all()):
            raise ValueError('All four valid ordered views and finite scene features required')
        # Boolean GPU indexing requires a dynamically sized nonzero result. The
        # contract is a fixed 4 x 9 x 16 layout, so check the stored mask when it
        # changes and extract the identical ordered channels with static slices.
        mask_key = (id(self.motion_mask), self.motion_mask._version,
                    self.motion_mask.device, self.motion_mask.dtype)
        if mask_key != self._validated_mask_key:
            expected = torch.zeros((4, 9, 16), dtype=torch.bool, device=scene.device)
            expected[:, 1:] = True
            if (self.motion_mask.dtype != torch.bool or self.motion_mask.device != scene.device
                    or not torch.equal(self.motion_mask, expected.flatten())):
                raise ValueError('Fixed ordered motion-channel mask required')
            self._validated_mask_key = mask_key
        normalized = ((scene - self.mean) / self.scale).clamp(-5, 5)
        motion = normalized.reshape(-1, 4, 9, 16)[:, :, 1:, :].reshape(-1, 512)
        if state is None:
            if elapsed is not None:
                raise ValueError('Episode start must have no elapsed interval')
            fast, slow = motion, motion
        else:
            if (not isinstance(state, tuple) or len(state) != 3
                    or any(not isinstance(s, torch.Tensor) or s.shape != motion.shape
                           or s.dtype != motion.dtype or s.device != motion.device
                           for s in state)
                    or isinstance(elapsed, bool) or not isinstance(elapsed, (float, int))
                    or not math.isfinite(elapsed) or not 0 < elapsed <= .08):
                raise ValueError('Prior traces and original positive interval up to 80ms required')
            if not (torch.isfinite(state[0]).all() & torch.isfinite(state[1]).all()
                    & torch.isfinite(state[2]).all()):
                raise ValueError('Prior traces and original positive interval up to 80ms required')
            fast_alpha = math.exp(-elapsed / self.fast_seconds)
            slow_alpha = math.exp(-elapsed / self.slow_seconds)
            # Only the previous observation was available during this interval.
            fast = fast_alpha * state[0] + (1 - fast_alpha) * state[2]
            slow = slow_alpha * state[1] + (1 - slow_alpha) * state[2]
        delta = .5 * torch.tanh(self.projection(fast - slow))
        return delta, (fast, slow, motion)

    def modulate(self, drive, vision_ports, delta):
        if (drive.ndim != 2 or len(drive) != len(delta)
                or delta.shape != (len(drive), self.projection.out_features)
                or vision_ports.ndim != 1 or len(vision_ports) != delta.shape[1]
                or not torch.isfinite(drive).all() or not torch.isfinite(delta).all()):
            raise ValueError('Original finite drive and matching visual ports required')
        result = drive.clone()
        result[:, vision_ports] = modulate_current(drive[:, vision_ports], delta)
        return result
