"""Six-muscle logit residual on the sealed four-muscle compact pedal head."""
import numpy as np
import torch
from torch import nn


def all_muscle_features(z, centers, radii):
    q = np.sum(((z[:, None, :] - centers) / radii[None, :, None])**2, axis=-1)
    return np.maximum(1-q, 0)**3


class CompactAllMuscleReadout(nn.Module):
    def __init__(self, compact, centers, radii, coefficients):
        super().__init__()
        n = len(centers)
        if (not 1 <= n <= 48 or centers.shape != (n, 6) or radii.shape != (n,)
                or coefficients.shape != (2, n)
                or any(x.dtype != torch.float32 or not x.isfinite().all()
                       for x in (centers, radii, coefficients))
                or (radii <= 0).any() or coefficients.abs().max() > 8):
            raise ValueError("Bounded compact six-muscle residual required")
        self.compact = compact
        for name, tensor in zip(("centers", "radii", "coefficients"), (centers, radii, coefficients)):
            self.register_buffer(name, tensor.detach().clone())

    def forward(self, muscles):
        compact = self.compact
        phase = compact.phase_head
        z = (muscles-phase.parent.base.training_reference)/phase.parent.base.training_scale
        steering = phase.parent(muscles)[:, :1]
        base = phase.pedal_projection(z)[:, 1:] + phase.offset(phase.features(muscles))
        prior_q = ((z[:, None, :4]-compact.centers)/compact.radii[None, :, None]).square().sum(-1)
        prior = (1-prior_q).clamp_min(0).pow(3) @ compact.coefficients.T
        q = ((z[:, None, :]-self.centers)/self.radii[None, :, None]).square().sum(-1)
        correction = (1-q).clamp_min(0).pow(3) @ self.coefficients.T
        return torch.cat((steering, (base+prior+correction).sigmoid()), dim=1)
