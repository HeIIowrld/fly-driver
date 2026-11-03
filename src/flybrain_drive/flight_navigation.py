"""Learned higher-visual cut boundary; same measured descending-to-wing motor edges."""

import torch
from torch import nn

from .flight_commands import COMMAND_GOAL_CONTRACT, SeparatedFlightCircuit


class SceneCommand(nn.Module):
    """Engineering missing-brain reduction, not measured DNa04/05 receptive fields."""

    def __init__(self, features, ports, *, baseline_current):
        super().__init__()
        if not 0 < baseline_current < .1:
            raise ValueError("Explicit small positive model command baseline required")
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("normalization_ready", torch.tensor(False))
        self.linear = nn.Linear(features, ports)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(torch.logit(torch.tensor(baseline_current)))

    @torch.no_grad()
    def set_train_normalization(self, mean, scale):
        if (mean.shape != self.mean.shape or scale.shape != self.scale.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or torch.any(scale <= 0)):
            raise ValueError("Finite training-only feature moments and positive scales required")
        self.mean.copy_(mean)
        self.scale.copy_(scale)
        self.normalization_ready.fill_(True)

    def forward(self, features, *, lesion=None):
        if lesion not in (None, "vision", "scene_command", "motion_channels", "intensity_channel",
                          "body", "goal"):
            raise ValueError("Unknown visual-command intervention")
        if not self.normalization_ready:
            raise ValueError("Training-only normalization must be sealed before scene commands")
        value = ((features - self.mean) / self.scale).clamp(-5, 5)
        if lesion in ("vision", "scene_command"):
            value = torch.zeros_like(value)
        elif lesion in ("motion_channels", "intensity_channel"):
            value = value.reshape(-1, 4, 9, 16).clone()
            if lesion == "motion_channels":
                value[:, :, 1:] = 0
            else:
                value[:, :, :1] = 0
            value = value.flatten(1)
        return self.linear(value).sigmoid()


class VisualFlightCircuit(SeparatedFlightCircuit):
    def __init__(self, spec_file, *, glutamate_sign, sparse_backend="cached_csr_gpu"):
        super().__init__(spec_file, glutamate_sign=glutamate_sign, sparse_backend=sparse_backend)
        if self.spec["schema"] != "flybrain.flight_circuit.visual_commands.v6":
            raise ValueError("Explicit visual-boundary v6 specification required")
        self.scene_command = SceneCommand(self.scene_pool.feature_count, len(self.steering_ports),
            baseline_current=self.spec["scene_command_baseline"])
        # First experiment isolates the added scene-to-DN adapter. Calibrated body stays a buffer.
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("scene_command."))

    @torch.no_grad()
    def import_v5(self, payload):
        if (payload.get("schema") != "flybrain.flight.separated_commands.v5"
                or payload.get("spec_sha256") != self.spec["source_v5_spec_sha256"]
                or payload.get("goal_contract") != COMMAND_GOAL_CONTRACT):
            raise ValueError("Explicit declared v5 source checkpoint required")
        state = self.state_dict()
        for key, value in payload["model"].items():
            if key not in state or value.shape != state[key].shape:
                raise ValueError("Unexpected v5 field: " + key)
            if key == "circuit.matrix":
                expected, source = state[key].coalesce().cpu(), value.coalesce().cpu()
                if (not torch.equal(expected.indices(), source.indices())
                        or not torch.equal(expected.values(), source.values())):
                    raise ValueError("Measured graph changed")
            state[key] = value
        self.load_state_dict(state, strict=True)
        self.assert_body_boundary_protected()

    def _drive_tensor(self, visual, visual_valid, body, body_valid, goals, scene_features,
                      *, lesion=None):
        parent_lesion = lesion if lesion in (None, "vision", "body", "goal") else None
        drive = super()._drive_tensor(visual, visual_valid, body, body_valid, goals,
                                      scene_features, lesion=parent_lesion)
        current = self.scene_command(scene_features, lesion=lesion)
        # Bounded summation of two nonnegative external currents; no direct motor residual.
        existing = drive[:, self.steering_ports]
        drive[:, self.steering_ports] = 1 - (1 - existing) * (1 - current)
        return drive
