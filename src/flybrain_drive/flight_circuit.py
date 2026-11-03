"""Measured flight-path topology with explicit engineering input and motor adapters.

This is an unvalidated rate reduction, not a spiking flight model. Neuromodulators,
electrical synapses, wingbeat timing and receptor-specific glutamate action are absent.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .episode_storage import load_array
from .graph import sha256
from .vehicle_circuit import SignedRateCircuit


def load_flight_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf8"))
    for field in ("graph", "annotations", "boundary", "transmitter", "scene_registration"):
        if sha256(spec[field + "_file"]) != spec[field + "_sha256"]:
            raise ValueError("Flight source changed: " + field)
    with load_array(spec["graph_file"], allow_pickle=False) as archive:
        graph = {key: archive[key] for key in archive.files}
    annotations = json.loads(Path(spec["annotations_file"]).read_text(encoding="utf8"))
    labels = json.loads(Path(spec["transmitter_file"]).read_text(encoding="utf8"))
    boundary = json.loads(Path(spec["boundary_file"]).read_text(encoding="utf8"))["boundary"]
    ids = graph["ids"].tolist()
    if [str(row["body_id"]) for row in boundary] != ids:
        raise ValueError("Boundary must match every measured neuron in order")
    by_id = {str(row["bodyId"]): row for row in annotations}
    nt = {str(row["body"]): row for row in labels}
    if set(by_id) != set(ids) or set(nt) != set(ids):
        raise ValueError("Every graph neuron requires original annotation/transmitter fields")
    declared = spec["vision_input_ids"] + spec["body_input_ids"] + spec["goal_input_ids"]
    if len(set(declared)) != len(declared) or not set(declared) <= set(ids):
        raise ValueError("Flight input groups must be distinct exact measured IDs")
    for body in spec["vision_input_ids"]:
        if not by_id[body]["type"].startswith(("HS", "VS")):
            raise ValueError("Visual cut-boundary port must retain its reviewed type")
    for body in spec["body_input_ids"]:
        row = by_id[body]
        if row["subclass"] != "haltere" or row["class"] != "mechanosensory_proprioceptive":
            raise ValueError("Body port must retain reviewed haltere class")
    for body in spec["goal_input_ids"]:
        if not by_id[body]["type"].startswith("DNg02"):
            raise ValueError("Task-goal port must be a declared DNg02 neuron")
    for group in ("b1", "b2", "power"):
        for side in ("L", "R"):
            for body in spec["motor_groups"][group][side]:
                row = by_id[body]
                expected = (
                    row["type"] == group + " MN"
                    if group != "power"
                    else row["type"].startswith(("DLMn", "DVMn"))
                )
                if not expected or row["somaSide"] != side:
                    raise ValueError("Flight motor type/side differs from source annotation")
    totals = np.array([row["incoming_synapses_total"] for row in boundary])
    return spec, graph, by_id, nt, totals


class FlightMotorReadout(nn.Module):
    """Output-only wing motor correspondence; it does not claim b1/b2 yaw physiology."""

    def __init__(self, output_ids, groups, *, power_reference):
        super().__init__()
        for group in ("b1", "b2", "power"):
            for side in ("L", "R"):
                indices = [output_ids.index(body) for body in groups[group][side]]
                if not indices or len(indices) != len(set(indices)):
                    raise ValueError("Nonempty distinct motor IDs required")
                self.register_buffer(group + "_" + side, torch.tensor(indices, dtype=torch.long))
        self.turn_log_gain = nn.Parameter(torch.tensor(2.0))
        self.forward_log_gain = nn.Parameter(torch.tensor(2.0))
        self.steering_mix = nn.Parameter(torch.zeros(2))
        self.register_buffer("power_reference", torch.tensor(float(power_reference)))
        self.register_buffer("lateral_reference", torch.zeros(2))
        self.register_buffer("reference_calibrated", torch.tensor(False))

    @torch.no_grad()
    def calibrate_reference(self, muscles, *, turn_scale, forward_scale):
        """Model-unit neutral/scale calibration, never measured muscle physiology."""
        muscles = torch.as_tensor(
            muscles, device=self.power_reference.device, dtype=self.power_reference.dtype
        )
        if (
            muscles.shape != (6,)
            or not torch.isfinite(muscles).all()
            or torch.any((muscles < 0) | (muscles > 1))
            or not np.isfinite([turn_scale, forward_scale]).all()
            or min(turn_scale, forward_scale) <= 0
        ):
            raise ValueError("Finite bounded model neutral and positive activity scales required")
        self.lateral_reference.copy_(
            torch.stack([muscles[1] - muscles[0], muscles[3] - muscles[2]])
        )
        self.power_reference.copy_(0.5 * (muscles[4] + muscles[5]))
        self.turn_log_gain.fill_(-np.log(turn_scale))
        self.forward_log_gain.fill_(-np.log(forward_scale))
        self.reference_calibrated.fill_(True)

    def forward(self, outputs):
        preactivation, muscles = self.preactivation(outputs)
        return preactivation.tanh(), muscles

    def preactivation(self, outputs):
        """Unbounded intent coordinate for stable inverse-tanh supervision, same forward codec."""
        b1_left = outputs[:, self.b1_L].mean(dim=1)
        b1_right = outputs[:, self.b1_R].mean(dim=1)
        b2_left = outputs[:, self.b2_L].mean(dim=1)
        b2_right = outputs[:, self.b2_R].mean(dim=1)
        power_left = outputs[:, self.power_L].mean(dim=1)
        power_right = outputs[:, self.power_R].mean(dim=1)
        lateral = torch.stack([b1_right - b1_left, b2_right - b2_left], dim=1)
        turn = self.turn_log_gain.exp() * (
            (lateral - self.lateral_reference) @ self.steering_mix.softmax(0)
        )
        common = 0.5 * (power_left + power_right)
        forward = self.forward_log_gain.exp() * (common - self.power_reference)
        intent = torch.stack([turn, forward], dim=1)
        muscles = torch.stack(
            [b1_left, b1_right, b2_left, b2_right, power_left, power_right], dim=1
        )
        return intent, muscles


class FlightScenePool(nn.Module):
    """Keep separate camera and neural-channel spatial summaries, not only global flow."""

    def __init__(self, uv, basis):
        super().__init__()
        uv = np.asarray(uv)
        if uv.shape != (721, 2) or not np.isfinite(uv).all():
            raise ValueError("Registered retinal image coordinates required")
        x, y = np.meshgrid(
            (np.arange(basis["columns"]) + 0.5) / basis["columns"],
            (np.arange(basis["rows"]) + 0.5) / basis["rows"],
        )
        centers = np.column_stack([x.ravel(), y.ravel()])
        weight = np.exp(
            -np.sum((centers[:, None] - uv[None]) ** 2, axis=-1)
            / (2 * basis["sigma_image_uv"] ** 2)
        )
        weight /= weight.sum(axis=1, keepdims=True)
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        self.feature_count = 4 * 9 * len(centers)

    def forward(self, scene):
        if scene.shape[-3:] != (4, 9, 721) or not torch.isfinite(scene).all():
            raise ValueError("Registered view/intensity/T4/T5/retinal scene required")
        return (torch.tanh(scene) @ self.weight.T).flatten(start_dim=1)


class FlightCircuit(nn.Module):
    """All sensory/goal effects must traverse original measured edges to wing MNs.

    Cut-boundary projection weights are trainable. Internal graph counts are fixed.
    Input targets are hypotheses, not certified visual or haltere receptor tuning.
    """

    def __init__(self, spec_file, *, glutamate_sign, sparse_backend="cached_csr_gpu"):
        super().__init__()
        if glutamate_sign not in (-1, 0, 1):
            raise ValueError("Declare a glutamate sign sensitivity hypothesis")
        spec, graph, annotated, nt, totals = load_flight_spec(spec_file)
        if (spec.get("schema") in ("flybrain.flight_circuit.separated_commands.v5",
                                   "flybrain.flight_circuit.visual_commands.v6")
                and type(self) is FlightCircuit):
            raise ValueError("SeparatedFlightCircuit or VisualFlightCircuit required")
        self.spec = spec
        self.spec_sha256 = sha256(spec_file)
        ids = graph["ids"].tolist()
        # Roles are declared external-drive ports; measured IDs/edges/counts do not change.
        graph["roles"] = graph["roles"].copy()
        for body in spec["goal_input_ids"]:
            graph["roles"][ids.index(body)] = "input"
        sign_map = {"acetylcholine": 1, "gaba": -1, "glutamate": glutamate_sign}
        signs = np.array([sign_map.get(nt[body]["consensus_nt"], 0) for body in ids])
        self.circuit = SignedRateCircuit(
            graph,
            signs,
            totals,
            dt=spec["dynamics"]["dt"],
            tau=spec["dynamics"]["tau"],
            sparse_backend=sparse_backend,
        )
        for group in ("vision", "body", "goal"):
            positions = [ids.index(body) for body in spec[group + "_input_ids"]]
            self.register_buffer(group + "_ports", torch.tensor(positions, dtype=torch.long))
        with load_array(spec["scene_registration_file"], allow_pickle=False) as registration:
            self.scene_pool = FlightScenePool(registration["image_uv"], spec["scene_basis"])
        self.vision = nn.Linear(12 + self.scene_pool.feature_count, len(spec["vision_input_ids"]))
        self.body = nn.Linear(6, len(spec["body_input_ids"]))
        self.goal = nn.Linear(2, len(spec["goal_input_ids"]))
        self.context = nn.Linear(20, len(spec["goal_input_ids"]), bias=False)
        with torch.no_grad():
            for layer in (self.vision, self.body, self.goal):
                nn.init.normal_(layer.weight, mean=0.0, std=0.05)
                layer.bias.fill_(-2.2)
            nn.init.normal_(self.context.weight, mean=0.0, std=0.05)
            # Initial bilateral command asymmetry is an explicit embodiment hypothesis.
            for index, body in enumerate(spec["goal_input_ids"]):
                self.goal.weight[index, 0] = 0.2 if annotated[body]["somaSide"] == "R" else -0.2
                self.goal.weight[index, 1] = 0.2
        self.register_buffer("visual_scale", torch.tensor([0.35] * 3 + [0.2] * 3))
        self.register_buffer(
            "body_scale",
            torch.tensor([0.1, 0.1, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0] + [1.0] * 4 + [100.0] * 4),
        )
        self.register_buffer("context_indices", torch.tensor([0, 1] + list(range(8, 16))))
        outputs = [body for body, role in zip(ids, graph["roles"], strict=True) if role == "output"]
        self.readout = FlightMotorReadout(
            outputs, spec["motor_groups"], power_reference=spec["embodiment"]["power_reference"]
        )
        self.neurons, self.glutamate_sign = len(ids), glutamate_sign
        self.sign_counts = {str(sign): int((signs == sign).sum()) for sign in (-1, 0, 1)}
        self.effective_edges = int(np.sum(signs[graph["pre"]] != 0))

    def initial_state(self, batch_size):
        return self.visual_scale.new_zeros((batch_size, self.neurons))

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
        batch = visual.shape[0]
        if (
            visual.shape != (batch, 6)
            or visual_valid.shape != visual.shape
            or body.shape != (batch, 16)
            or body_valid.shape != body.shape
            or goals.shape != (batch, 2)
            or state.shape != (batch, self.neurons)
            or scene.shape != (batch, 4, 9, 721)
            or scene_valid.shape != (batch, 4)
            or visual_valid.dtype != torch.bool
            or body_valid.dtype != torch.bool
            or scene_valid.dtype != torch.bool
            or not all(torch.isfinite(value).all() for value in (visual, body, goals))
            or torch.any(goals.abs() > 1)
        ):
            raise ValueError("Versioned finite visual/body/mask/task-goal inputs required")
        if not scene_valid.all() or not body_valid[:, [0, 1, 4]].all():
            raise ValueError("Missing required visual/attitude/yaw feedback; no silent fallback")
        if lesion not in (None, "vision", "body", "goal"):
            raise ValueError("Explicit input lesion identity required")
        drive = self._drive_tensor(
            visual, visual_valid, body, body_valid, goals, self.scene_pool(scene), lesion=lesion
        )
        outputs, state = self.circuit(drive, state)
        intent, muscles = self.readout(outputs)
        return intent, muscles, state

    def _drive_tensor(
        self, visual, visual_valid, body, body_valid, goals, scene_features, *, lesion=None
    ):
        """Sealed training cache only; public forward validates full scene and sensor contracts."""
        drive = body.new_zeros((len(body), self.neurons))
        if lesion != "vision":
            value = torch.cat(
                [
                    torch.tanh(visual / self.visual_scale) * visual_valid,
                    visual_valid.float(),
                    scene_features,
                ],
                dim=1,
            )
            drive[:, self.vision_ports] = self.vision(value).sigmoid()
        if lesion != "body":
            value = torch.cat(
                [torch.tanh(body[:, 2:5]) * body_valid[:, 2:5], body_valid[:, 2:5].float()], dim=1
            )
            drive[:, self.body_ports] = self.body(value).sigmoid()
        if lesion != "goal":
            descending = self.goal(goals)
            if lesion != "body":
                normalized = torch.tanh(body / self.body_scale) * body_valid
                context = torch.cat(
                    [
                        normalized[:, self.context_indices],
                        body_valid[:, self.context_indices].float(),
                    ],
                    dim=1,
                )
                descending = descending + self.context(context)
            drive[:, self.goal_ports] = descending.sigmoid()
        return drive

    def _advance_elapsed_tensor(self, drive, state, elapsed):
        """Preserve fractional original intervals under the declared exponential rate step.

        Assumes an unlesioned state mask; input/edge lesions remain supported. No
        state or input sample is reset or invented by the internal integration.
        """
        nominal = self.spec["dynamics"]["dt"]
        if not np.isfinite(elapsed) or not 0 < elapsed <= nominal + 1e-12:
            raise ValueError("Positive original subinterval at most nominal dt required")
        _, next_state = self.circuit._advance_tensor(drive, state)
        nominal_alpha = float(np.float32(-np.expm1(-nominal / self.spec["dynamics"]["tau"])))
        ratio = -np.expm1(-elapsed / self.spec["dynamics"]["tau"]) / nominal_alpha
        if abs(ratio - 1) > 1e-7:
            next_state = state + ratio * (next_state - state)
        return next_state

    def _read_state_tensor(self, state):
        return self.readout(state[:, self.circuit.output_ports])

    def _read_state_preactivation_tensor(self, state):
        return self.readout.preactivation(state[:, self.circuit.output_ports])
