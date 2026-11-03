"""Separated descending command ports with an explicitly frozen virtual-body boundary."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from .flight_circuit import FlightCircuit, load_flight_spec
from .flight_learning import GOAL_CONTRACT
from .graph import sha256

COMMAND_GOAL_CONTRACT = {
    **GOAL_CONTRACT,
    "schema": "flybrain.flight.speed_goal.v2",
    "biological_claim": ("engineering split: turn bias to DNa04/05, speed to DNg02; "
                         "not measured tuning"),
    "checkpoint_compatibility": ("explicit calibrated-v4 conversion only; "
                                 "numerical goal fields retained"),
}


class FrozenLinear(nn.Module):
    """Buffers stay outside optimizer parameter groups, including momentum/weight decay."""

    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("bias", bias.detach().clone())

    def forward(self, value):
        return functional.linear(value, self.weight, self.bias)


class PowerGoal(nn.Linear):
    """Read the desired-speed coordinate only; never the lateral task bias."""

    def forward(self, goals):
        return functional.linear(goals[:, 1:2], self.weight, self.bias)


class SteeringCommand(nn.Module):
    """Nonnegative neural currents in experimentally oriented existing DN IDs."""

    def __init__(self, orientations):
        super().__init__()
        self.register_buffer("orientations", torch.tensor(orientations, dtype=torch.float32))
        self.gain_logits = nn.Parameter(torch.full((len(orientations),), -1.38629436112))

    def forward(self, goals):
        return self.gain_logits.sigmoid() * (goals[:, :1] * self.orientations).relu()


class SeparatedFlightCircuit(FlightCircuit):
    """Same measured edges; distinct command boundaries, no direct command-to-motor bypass."""

    def __init__(self, spec_file, *, glutamate_sign, sparse_backend="cached_csr_gpu"):
        spec, graph, annotations, _, _ = load_flight_spec(spec_file)
        schema = spec.get("schema")
        if schema not in ("flybrain.flight_circuit.separated_commands.v5",
                          "flybrain.flight_circuit.visual_commands.v6"):
            raise ValueError("Explicit v5 separated command specification required")
        if (schema == "flybrain.flight_circuit.visual_commands.v6"
                and type(self) is SeparatedFlightCircuit):
            raise ValueError("VisualFlightCircuit required for v6; no implicit v5 routing")
        if spec.get("goal_contract") != COMMAND_GOAL_CONTRACT:
            raise ValueError("Explicit separated-goal semantics required")
        steering = spec["steering_input_ids"]
        if (len(steering) != 4 or len(set(steering)) != 4
                or any(annotations[bid]["type"] not in ("DNa04", "DNa05") for bid in steering)
                or set(steering) & set(spec["vision_input_ids"] + spec["body_input_ids"]
                                      + spec["goal_input_ids"])):
            raise ValueError("Four distinct measured DNa04/05 ports required")
        evidence = Path(spec["steering_orientation_evidence"])
        if sha256(evidence) != spec["steering_orientation_evidence_sha256"]:
            raise ValueError("Engineering steering orientation evidence changed")
        rows = json.loads(evidence.read_text(encoding="utf8"))
        by_id = {row["injected_ids"][0]: row for row in rows if len(row["injected_ids"]) == 1}
        orientations = spec["steering_orientations"]
        if (len(orientations) != 4 or any(sign not in (-1, 1) for sign in orientations)
                or any(by_id[bid]["current_readout_delta"][0] * sign <= .01
                       for bid, sign in zip(steering, orientations, strict=True))):
            raise ValueError("Steering orientation differs from declared model injection response")
        super().__init__(spec_file, glutamate_sign=glutamate_sign, sparse_backend=sparse_backend)
        positions = torch.tensor([graph["ids"].tolist().index(bid) for bid in steering])
        self.register_buffer("steering_ports", positions)
        self.circuit.input_mask[positions] = True
        self.goal = PowerGoal(1, len(spec["goal_input_ids"]))
        self.steering_command = SteeringCommand(orientations)
        self.body = FrozenLinear(self.body.weight, self.body.bias)
        self.register_buffer("protected_body_weight", self.body.weight.clone())
        self.register_buffer("protected_body_bias", self.body.bias.clone())
        self.readout.requires_grad_(False)

    def _drive_tensor(self, visual, visual_valid, body, body_valid, goals, scene_features,
                      *, lesion=None):
        drive = super()._drive_tensor(visual, visual_valid, body, body_valid, goals,
                                      scene_features, lesion=lesion)
        if lesion != "goal":
            drive[:, self.steering_ports] = self.steering_command(goals)
        return drive

    def assert_body_boundary_protected(self):
        if (not torch.equal(self.body.weight, self.protected_body_weight)
                or not torch.equal(self.body.bias, self.protected_body_bias)
                or list(self.body.parameters())):
            raise ValueError("Frozen calibrated body boundary was changed")

    @torch.no_grad()
    def import_calibrated_v4(self, payload):
        """Explicit conversion; preserve zero-turn outputs, remove lateral input from power DN."""
        if payload.get("schema") != "flybrain.flight.speed_yaw_calibration.v1":
            raise ValueError("Explicitly yaw-calibrated v4 source required")
        if (payload.get("goal_contract") != GOAL_CONTRACT
                or payload.get("spec_sha256") != self.spec["source_v4_spec_sha256"]):
            raise ValueError("Declared v4 source spec/goal semantics differ")
        source = payload["model"]
        own = self.state_dict()
        for key, value in source.items():
            if key == "goal.weight":
                own[key] = value[:, 1:2]
            elif key == "circuit.input_mask":
                expected = value.clone()
                expected[self.steering_ports.cpu()] = True
                if not torch.equal(expected.cpu(), self.circuit.input_mask.cpu()):
                    raise ValueError("Unexpected role change beyond four new command ports")
            elif key == "circuit.matrix":
                matrix = self.circuit.matrix.coalesce().cpu()
                if (not torch.equal(matrix.indices(), value.coalesce().indices())
                        or not torch.equal(matrix.values(), value.coalesce().values())):
                    raise ValueError("Original graph/sign normalization changed")
            elif key not in own or own[key].shape != value.shape:
                raise ValueError("Unexpected v4 conversion field: " + key)
            else:
                own[key] = value
        own["protected_body_weight"] = source["body.weight"].clone()
        own["protected_body_bias"] = source["body.bias"].clone()
        self.load_state_dict(own, strict=True)
        self.assert_body_boundary_protected()
