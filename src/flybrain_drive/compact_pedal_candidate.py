"""Restore the sealed compact pedal continuation into its unchanged phase actor."""
from pathlib import Path

import torch

from .balanced_muscle_candidate import load_balanced_pedal_runtime
from .compact_all_muscle_readout import CompactAllMuscleReadout
from .compact_flight_pedal_readout import CompactFlightPedalReadout
from .graph import sha256
from .sac_unified_student import read


def load_compact_pedal_runtime(actor_kind, checkpoint, expected_sha256):
    path = Path(checkpoint)
    result = read(path.parent / "result.json")
    config = read(path.parent / "manifest.json")["config"]
    if (result["status"] != "completed_compact_braking_fit" or result["optimizer_success"] is not True
            or result["actor_qualified"] is not False or result["anchor_max_difference"] != 0
            or sha256(path) != expected_sha256 or result["checkpoint_sha256"] != expected_sha256
            or config["development_used_for_fit"] is not False
            or sha256(Path(__file__).with_name("compact_flight_pedal_readout.py")) != config["compact_readout_sha256"]):
        raise ValueError("Exact completed TRAIN-only compact fit required")
    cp = torch.load(path, map_location="cpu", weights_only=True)
    phase = Path(config["arguments"]["phase_fit"]) / "pedals.pt"
    residual = cp.get("schema") == "flybrain.compact_flight_pedals.experimental.v2"
    keys = {"schema", "phase_checkpoint", "phase_checkpoint_sha256", "centers",
            "radii", "coefficients", "actor_kind", "actor_qualified"}
    if residual:
        keys.add("prior")
    if (set(cp) != keys
            or cp["schema"] not in ("flybrain.compact_flight_pedals.experimental.v1",
                                    "flybrain.compact_flight_pedals.experimental.v2")
            or (residual and (set(cp["prior"]) != {"centers","radii","coefficients"}
                or sha256(Path(__file__).with_name("compact_all_muscle_readout.py")) != config.get("all_muscle_readout_sha256")))
            or cp["actor_qualified"] is not False or cp["actor_kind"] not in ("parent", "dual_guard")
            or actor_kind != "compact_" + cp["actor_kind"] or phase != Path(cp["phase_checkpoint"])
            or cp["phase_checkpoint_sha256"] != config["phase_checkpoint_sha256"]
            or read(phase.parent / "manifest.json")["config"]["actor_kind"] != cp["actor_kind"]):
        raise ValueError("Exact compact checkpoint and phase lineage required")
    if residual:
        prior_path = Path(config["arguments"]["protect_compact"]) / "pedals.pt"
        if sha256(prior_path) != config["protected_checkpoint_sha256"]:
            raise ValueError("Exact protected compact checkpoint required")
        prior = torch.load(prior_path, map_location="cpu", weights_only=True)
        if (prior["schema"] != "flybrain.compact_flight_pedals.experimental.v1"
                or prior["actor_kind"] != cp["actor_kind"]
                or prior["phase_checkpoint_sha256"] != cp["phase_checkpoint_sha256"]
                or not all(torch.equal(prior[key], value) for key, value in cp["prior"].items())):
            raise ValueError("Embedded protected compact tensors differ")
    runtime, receipt = load_balanced_pedal_runtime(phase, cp["phase_checkpoint_sha256"])
    try:
        base = cp["prior"] if residual else cp
        head = CompactFlightPedalReadout(runtime.head, base["centers"], base["radii"], base["coefficients"])
        if residual:
            head = CompactAllMuscleReadout(head, cp["centers"], cp["radii"], cp["coefficients"])
        runtime.head = head.eval().requires_grad_(False)
        return runtime, dict(actor_kind=actor_kind, checkpoint_sha256=expected_sha256,
                             phase_receipt=receipt, actor_qualified=False, vehicle_control_sent=False)
    except BaseException:
        runtime.close()
        raise
