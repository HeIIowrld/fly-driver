"""Replay a diagnostic brake pulse through a frozen body adapter, without fitting."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_body_feedback_adapter import VehicleBodyFeedbackAdapter
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit, candidate = Path(args.audit), Path(args.candidate)
    result, manifest = read(audit / "result.json"), read(audit / "manifest.json")
    config = manifest["config"]
    if (
        result["status"] != "completed_independent_native_brake_response_audit"
        or result["fit_allowed"] is not False
        or not result["native_response_observed"]
    ):
        raise ValueError("Completed diagnostic-only original brake audit required")
    root = Path(config["arguments"]["recording"])
    samples_path = root / "control/native-motion-samples.json"
    if sha256(samples_path) != config["native_samples_sha256"]:
        raise ValueError("Original native samples changed")
    candidate_result = read(candidate / "result.json")
    checkpoint = candidate / "body-feedback-adapter.pt"
    if (
        candidate_result["status"] != "completed_brake_steering_anchored_body_fit"
        or sha256(checkpoint) != candidate_result["checkpoint_sha256"]
    ):
        raise ValueError("Exact completed original adapter required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["schema"] != "flybrain.experimental_body_feedback_adapter.v1":
        raise ValueError("Original experimental adapter schema required")
    adapter = VehicleBodyFeedbackAdapter(payload["model"]["projection.weight"].shape[0])
    adapter.load_state_dict(payload["model"], strict=True)
    adapter.eval().requires_grad_(False)
    contract = "configs/ac-virtual-navigation-body-v2.json"
    identity = read("configs/ac-virtual-body-v1.json")["native_identity"]
    encoder = NavigationBodyEncoder(contract, native_identity=identity)
    low, high = result["pulse_host_receipt_bounds"]
    observations, seen = [], set()
    for row in read(samples_path):
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Native prefix hash differs")
        time = row["received_monotonic_seconds"]
        native = decode_physics_prefix(raw, time)
        if native.packet_id != row["packet_id"]:
            raise ValueError("Native packet differs")
        if native.packet_id in seen:
            continue
        seen.add(native.packet_id)
        body = encoder.encode(
            decode_navigation_body_prefix(raw, time),
            episode_id="brake-diagnostic",
            decision_monotonic_seconds=time,
        )
        observations.append((time, native.fields, body))
    before = [x for x in observations if low - 0.5 <= x[0] < low]
    braking = [
        x
        for x in observations
        if high <= x[0] <= low + 1.5 and x[1]["brake"] > 0.5 and x[1]["speedKmh"] > 1
    ]
    if not before or not braking or not all(x[2].valid[8:12].all() for x in before + braking):
        raise ValueError("Original bounded before/braking samples with valid load required")
    reference = np.mean([x[2].values[8:12] for x in before], axis=0)
    bodies = np.stack([x[2].values for x in braking]).astype(np.float32)
    masks = np.stack([x[2].valid for x in braking])
    changed = bodies.copy()
    changed[:, 8:12] = reference
    with torch.no_grad():
        original = adapter(torch.from_numpy(bodies), torch.from_numpy(masks)).numpy()
        counterfactual = adapter(torch.from_numpy(changed), torch.from_numpy(masks)).numpy()
    delta = original - counterfactual
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "native_samples_sha256": sha256(samples_path),
            "source_audit_sha256": sha256(audit / "result.json"),
            "candidate_checkpoint_sha256": sha256(checkpoint),
            "role": "diagnostic only; fitting prohibited",
            "counterfactual": "replace only four load fractions with pre-pulse mean",
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    output = {
        "status": "completed_brake_load_adapter_diagnostic",
        "before_samples": len(before),
        "moving_braking_samples": len(braking),
        "before_front_load_fraction_mean": float(reference[:2].sum()),
        "braking_front_load_fraction_mean": float(bodies[:, 8:10].sum(axis=1).mean()),
        "adapter_output_mean_absolute_change": float(np.abs(delta).mean()),
        "adapter_output_max_absolute_change": float(np.abs(delta).max()),
        "adapter_outputs": int(delta.shape[1]),
        "fit_allowed": False,
        "optimizer_updates": 0,
        "actual_wind_used": False,
        "full_graph_replayed": False,
        "actor_deployed": False,
        "model_effect_or_vehicle_control_validated": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
