"""Engineering auxiliary supervision on existing flight muscle state only.

Future teacher controls are targets to ``masked_loss``, never forward inputs.
This head supplies no vehicle command and does not change measured connections.
"""
import torch
from torch import nn


class FlightAnticipationReadout(nn.Module):
    def __init__(self, reference, scale, horizon_count=4, scale_floor=1e-6):
        super().__init__()
        if (not isinstance(reference, torch.Tensor) or reference.ndim != 1
                or not isinstance(scale, torch.Tensor) or scale.shape != reference.shape
                or reference.dtype != scale.dtype or reference.device != scale.device
                or not reference.is_floating_point() or not torch.isfinite(reference).all()
                or not torch.isfinite(scale).all() or torch.any(scale < 0)
                or type(horizon_count) is not int or horizon_count < 1
                or not 0 < scale_floor < 1):
            raise ValueError('Finite train-only muscle calibration and horizon count required')
        self.register_buffer('reference', reference.detach().clone())
        self.register_buffer('scale', scale.detach().clone().clamp_min(scale_floor))
        self.horizon_count = horizon_count
        self.projection = nn.Linear(len(reference), horizon_count * 3,
                                    device=reference.device, dtype=reference.dtype)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, muscles):
        if (not isinstance(muscles, torch.Tensor) or muscles.ndim != 2
                or muscles.shape[1] != len(self.reference) or len(muscles) < 1
                or muscles.device != self.reference.device or muscles.dtype != self.reference.dtype
                or not torch.isfinite(muscles).all()):
            raise ValueError('Existing finite flight muscle activities required')
        return self.projection((muscles - self.reference) / self.scale).reshape(
            len(muscles), self.horizon_count, 3)

    def masked_loss(self, predictions, targets, valid):
        if (predictions.ndim != 3 or predictions.shape[1:] != (self.horizon_count, 3)
                or targets.shape != predictions.shape or valid.shape != predictions.shape[:2]
                or valid.dtype != torch.bool or valid.device != predictions.device
                or targets.device != predictions.device
                or not torch.isfinite(predictions).all()
                or not torch.isfinite(targets[valid]).all()):
            raise ValueError('Explicit future target horizon masks required')
        if not valid.any():
            return predictions.sum() * 0
        # Mask BEFORE arithmetic: absent targets, including NaN, have no influence.
        error = predictions[valid] - targets.detach()[valid]
        return (error.square() * error.new_tensor([4., 1., 1.])).mean()
