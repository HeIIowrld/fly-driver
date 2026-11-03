"""Engineering body baseline plus visual residual on registered FlyVis features.

The first 896 features are the original six retinal and eight T4/T5 pooled
responses. The final 128 geometry/support bits are excluded from the visual
residual so that they cannot masquerade as an image contribution. This readout
is neither measured fly anatomy nor a qualified actuator policy.
"""

from torch import nn


class BodyPreview(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(42, 16)
        self.hidden = nn.Linear(16, 64)
        self.output = nn.Linear(64, 6)

    def forward(self, body):
        if body.ndim != 2 or body.shape[1] != 42:
            raise ValueError('Original masked body21 features required')
        return self.output(self.hidden(self.input(body).tanh()).tanh())


class ResidualPreview(nn.Module):
    """Predict original six standardized teacher-path preview heads."""

    def __init__(self):
        super().__init__()
        self.body = BodyPreview()
        self.vision = nn.Linear(896, 6, bias=False)
        nn.init.zeros_(self.vision.weight)

    def forward(self, visual, body):
        if visual.ndim != 2 or visual.shape[1] != 1024 or len(visual) != len(body):
            raise ValueError('Original registered FlyVis features required')
        return self.body(body) + self.vision(visual[:, :896])
