"""Engineering higher-brain boundary into fixed fly descending and motor circuits."""

import json
from pathlib import Path

import torch
from torch import nn

from .flight_commands import COMMAND_GOAL_CONTRACT
from .flight_navigation import VisualFlightCircuit
from .graph import sha256


class NavigationCommandBoundary(nn.Module):
    """Retinotopic nonlinear command modulation, not measured higher-brain anatomy.

    Turning learns from vision/task plus speed magnitude, never yaw-as-label
    shortcut. Protected haltere feedback remains in the inherited circuit drive.
    Power additionally receives the explicitly declared virtual-body state.
    """

    def __init__(self, scene_mean, scene_scale):
        super().__init__()
        self.register_buffer("mean", scene_mean.detach().clone())
        self.register_buffer("scale", scene_scale.detach().clone())
        self.register_buffer(
            "body_scale",
            torch.tensor(
                [
                    0.1,
                    0.1,
                    1,
                    1,
                    1,
                    10,
                    10,
                    10,
                    1,
                    1,
                    1,
                    1,
                    100,
                    100,
                    100,
                    100,
                    30 / 3.6,
                    30,
                    30,
                    30,
                    30,
                ]
            ),
        )
        self.patch = nn.Linear(9, 8)
        self.turn_hidden = nn.Linear(512 + 2 + 2, 64)
        self.power_hidden = nn.Linear(512 + 42 + 2, 32)
        self.turn_output = nn.Linear(64, 4)
        self.power_output = nn.Linear(32, 1)
        for layer in (self.turn_output, self.power_output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, scene_features, body, body_valid, goals, *, lesion=None):
        if lesion not in (None, "vision", "body", "goal", "new_command"):
            raise ValueError("Unknown navigation boundary lesion")
        scene = ((scene_features - self.mean) / self.scale).clamp(-5, 5)
        if lesion == "vision":
            scene = torch.zeros_like(scene)
        patches = scene.reshape(-1, 4, 9, 16).transpose(2, 3)
        visual = self.patch(patches).tanh().flatten(1)
        normalized = torch.tanh(body / self.body_scale) * body_valid
        masks = body_valid.float()
        if lesion == "body":
            normalized, masks = torch.zeros_like(normalized), torch.zeros_like(masks)
        if lesion == "goal":
            goals = torch.zeros_like(goals)
        # Differential wheel rates can proxy current yaw; keep them out of navigation turning.
        locomotion = torch.cat([normalized[:, 16:17], masks[:, 16:17]], 1)
        turn = self.turn_output(self.turn_hidden(torch.cat([visual, locomotion, goals], 1)).tanh())
        power = self.power_output(
            self.power_hidden(torch.cat([visual, normalized, masks, goals], 1)).tanh()
        )
        delta = 8 * torch.tanh(torch.cat([turn, power], 1) / 8)
        return torch.zeros_like(delta) if lesion == "new_command" else delta


def modulate_current(current, log_odds_delta):
    """Positive bounded relative-current modulation; delta0 preserves current exactly."""
    factor = log_odds_delta.exp()
    return current * factor / (1 - current + current * factor)


class EmbodiedVisualFlightCircuit(nn.Module):
    schema = "flybrain.flight.embodied_navigation.v7"

    def __init__(
        self,
        source_spec,
        source_payload,
        *,
        sparse_backend="cached_csr_gpu",
        body_contract="configs/ac-virtual-navigation-body-v2.json",
    ):
        super().__init__()
        if (
            source_payload.get("schema") != "flybrain.flight.visual_commands.v6"
            or source_payload.get("spec_sha256") != sha256(source_spec)
            or source_payload.get("goal_contract") != COMMAND_GOAL_CONTRACT
        ):
            raise ValueError("Explicit v6 checkpoint required; no implicit conversion")
        self.backbone = VisualFlightCircuit(
            source_spec,
            glutamate_sign=source_payload["glutamate_sign"],
            sparse_backend=sparse_backend,
        )
        self.backbone.load_state_dict(source_payload["model"], strict=True)
        if not self.backbone.scene_command.normalization_ready:
            raise ValueError("Sealed TRAIN-only source scene normalization required")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.boundary = NavigationCommandBoundary(
            self.backbone.scene_command.mean, self.backbone.scene_command.scale
        )
        contract = json.loads(Path(body_contract).read_text(encoding="utf8"))
        if (
            contract["schema"] != "flybrain.virtual_navigation_body.contract.v2"
            or contract["feature_count"] != 21
            or sha256(contract["legacy_contract"]) != contract["legacy_contract_sha256"]
        ):
            raise ValueError("Explicit vehicle feedback21 contract required")
        self.spec = dict(
            self.backbone.spec,
            schema="flybrain.flight_circuit.embodied_navigation.v7",
            source_visual_spec=str(source_spec),
            source_visual_spec_sha256=sha256(source_spec),
            body_contract=str(body_contract),
            body_contract_sha256=sha256(body_contract),
            body_feature_count=21,
            learned_boundary="engineering nonlinear retinotopic DN modulation",
        )

    def forward(
        self,
        visual,
        visual_valid,
        body,
        body_valid,
        goals,
        state,
        *,
        scene,
        scene_valid,
        lesion=None,
    ):
        batch = len(body)
        if (
            visual.shape != (batch, 6)
            or visual_valid.shape != visual.shape
            or visual_valid.dtype != torch.bool
            or goals.shape != (batch, 2)
            or state.shape != (batch, self.backbone.neurons)
            or scene.shape != (batch, 4, 9, 721)
            or scene_valid.shape != (batch, 4)
            or scene_valid.dtype != torch.bool
            or not scene_valid.all()
            or not all(torch.isfinite(x).all() for x in (visual, goals, state, scene))
            or torch.any(goals.abs() > 1)
        ):
            raise ValueError("Explicit finite scene/body21/task/state contracts required")
        drive = self._drive_tensor(
            visual,
            visual_valid,
            body,
            body_valid,
            goals,
            self.backbone.scene_pool(scene),
            lesion=lesion,
        )
        outputs, state = self.backbone.circuit(drive, state)
        intent, muscles = self.backbone.readout(outputs)
        return intent, muscles, state

    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

    def _drive_tensor(
        self, visual, visual_valid, body, body_valid, goals, scene_features, *, lesion=None
    ):
        if (
            body.shape != (len(body), 21)
            or body_valid.shape != body.shape
            or body_valid.dtype != torch.bool
            or not torch.isfinite(body).all()
            or not body_valid[:, [0, 1, 4, 16, 17, 18, 19, 20]].all()
        ):
            raise ValueError("New21-feature body contract and valid vehicle feedback required")
        legacy_lesion = lesion if lesion in ("vision", "body", "goal") else None
        drive = self.backbone._drive_tensor(
            visual,
            visual_valid,
            body[:, :16],
            body_valid[:, :16],
            goals,
            scene_features,
            lesion=legacy_lesion,
        )
        return self._modulate_cached_drive(
            drive, body, body_valid, goals, scene_features, lesion=lesion
        )

    def _modulate_cached_drive(
        self, baseline, body, body_valid, goals, scene_features, *, lesion=None
    ):
        """Sealed offline cache only; inherited feedforward weights must remain frozen."""
        drive = baseline.clone()
        delta = self.boundary(scene_features, body, body_valid, goals, lesion=lesion)
        steering, power = self.backbone.steering_ports, self.backbone.goal_ports
        drive[:, steering] = modulate_current(drive[:, steering], delta[:, :4])
        drive[:, power] = modulate_current(drive[:, power], delta[:, 4:5])
        return drive

    def _advance_elapsed_tensor(self, drive, state, elapsed):
        return self.backbone._advance_elapsed_tensor(drive, state, elapsed)

    def _read_state_tensor(self, state):
        return self.backbone._read_state_tensor(state)

    def _read_state_preactivation_tensor(self, state):
        return self.backbone._read_state_preactivation_tensor(state)
