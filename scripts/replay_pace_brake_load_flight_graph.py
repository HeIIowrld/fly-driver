"""Replay a held diagnostic pace history through the frozen flight body graph."""

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
from flybrain_drive.flight_embodied_navigation import modulate_current
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "recording", "candidate", "fit-plan", "fit-result",
                 "pedal-checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    audit, candidate = Path(args.audit), Path(args.candidate)
    proof, config = read(audit / "result.json"), read(audit / "manifest.json")["config"]
    if (proof["status"] != "completed_pace_brake_diagnostic_audit"
            or proof["training_admissible"] is not False):
        raise ValueError("Diagnostic-only pace braking audit required")
    source = Path(args.recording)
    samples_path = source / "events.jsonl"
    if sha256(samples_path) != config["events_sha256"]:
        raise ValueError("Original events hash differs")
    brake_rows = read(audit / "moving-brake-receipts.json")
    if sha256(audit / "moving-brake-receipts.json") != proof["receipt_rows_sha256"]:
        raise ValueError("Brake receipt rows changed")
    low = high = brake_rows[0]["native_receipt"]
    fit, plan, pedal = read(args.fit_result), read(args.fit_plan), read(args.pedal_checkpoint)
    candidate_result = read(candidate / "result.json")
    ckpt = candidate / "body-feedback-adapter.pt"
    if (
        fit["status"] != "completed_one_pass_pedal_component_fit"
        or candidate_result["status"] != "completed_brake_steering_anchored_body_fit"
        or sha256(ckpt) != candidate_result["checkpoint_sha256"]
        or pedal["bindings"] != fit["bindings"]
    ):
        raise ValueError("Exact original fit and candidate required")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    if payload["source_pedal_checkpoint_sha256"] != sha256(args.pedal_checkpoint):
        raise ValueError("Adapter and pedal candidate binding differs")
    encoder = NavigationBodyEncoder(
        "configs/ac-virtual-navigation-body-v2.json",
        native_identity=read("configs/ac-virtual-body-v1.json")["native_identity"],
    )
    rows, seen = [], set()
    for line in samples_path.read_text(encoding="utf8").splitlines():
        event = json.loads(line)
        row = event["native"]
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
            raise ValueError("Native source prefix changed")
        time = row["received_monotonic_seconds"]
        native = decode_physics_prefix(raw, time)
        if native.packet_id != row["packet_id"]:
            raise ValueError("Original packet differs")
        if native.packet_id in seen:
            continue
        seen.add(native.packet_id)
        body = encoder.encode(
            decode_navigation_body_prefix(raw, time),
            episode_id="brake-diagnostic",
            decision_monotonic_seconds=time,
        )
        if low - 0.5 <= time <= low + 1.5:
            rows.append((time, native.fields, body))
    before = [x[2].values[8:12] for x in rows if x[0] < low]
    if (
        not before
        or len(rows) < 2
        or not all(x[2].valid[[4, 8, 9, 10, 11, 16]].all() for x in rows)
    ):
        raise ValueError("Valid original body samples and pre-pulse reference required")
    times = np.array([r[0] for r in rows])
    if np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 0.08):
        raise ValueError("Original causal interval must be positive and at most 80ms")
    reference = np.mean(before, axis=0)
    body = np.stack([x[2].values for x in rows]).astype(np.float32)
    valid = np.stack([x[2].valid for x in rows])
    changed = body.copy()
    changed[:, 8:12] = reference
    base = OriginalGraphAdapter(plan, lambda: None)
    with torch.no_grad():
        for name, parameter in base.parameters().items():
            parameter.copy_(torch.tensor(pedal["values"][name], device=parameter.device))
        quiet = base.initialize()
    if base.fixed_fingerprint() != fit["fit"]["final_fixed_rollout"]["fixed_fingerprint"]:
        raise ValueError("Original graph/readout fingerprint differs")
    base.model.eval().requires_grad_(False)
    adapter = VehicleBodyFeedbackAdapter(len(base.model.backbone.body_ports)).to("cuda")
    adapter.load_state_dict(payload["model"], strict=True)
    adapter.eval().requires_grad_(False)
    b = torch.tensor(body, device="cuda")
    v = torch.tensor(valid, device="cuda")
    alternate = torch.tensor(changed, device="cuda")
    command = torch.tensor([[0.0, 30.0 / 3.6]], device="cuda").expand(len(rows), -1)
    goals = torch.tensor([[0.0, 1.0]], device="cuda").expand(len(rows), -1)
    with torch.no_grad():
        fixed_drive = base.model.drive(command, b, v, goals, **base.kwargs)
        ports = base.model.backbone.body_ports
        actual, counterfactual = fixed_drive.clone(), fixed_drive.clone()
        actual[:, ports] = modulate_current(fixed_drive[:, ports], adapter(b, v))
        counterfactual[:, ports] = modulate_current(fixed_drive[:, ports], adapter(alternate, v))
        state = quiet.expand(2, -1).clone()
        outputs = []
        for index in range(len(rows)):
            if index:
                drive = torch.stack((actual[index - 1], counterfactual[index - 1]))
                elapsed = float(times[index] - times[index - 1])
                dt = base.model.backbone.spec["dynamics"]["dt"]
                count = int(np.floor(elapsed / dt))
                for _ in range(count):
                    state = base.model.backbone._advance_elapsed_tensor(drive, state, dt)
                tail = elapsed - count * dt
                if tail > 1e-12:
                    state = base.model.backbone._advance_elapsed_tensor(drive, state, tail)
            outputs.append(base.predict(state)["canonical"].cpu().numpy())
    outputs = np.stack(outputs)
    selection = np.array(
        [t >= high and fields["brake"] > 0.5 and fields["speedKmh"] > 1 for t, fields, _ in rows]
    )
    if not selection.any() or not np.isfinite(outputs).all():
        raise ValueError("Finite braking outputs required")
    delta = outputs[:, 0] - outputs[:, 1]
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "native_samples_sha256": sha256(samples_path),
            "source_audit_sha256": sha256(audit / "result.json"),
            "fit_plan_sha256": sha256(args.fit_plan),
            "fit_result_sha256": sha256(args.fit_result),
            "pedal_checkpoint_sha256": sha256(args.pedal_checkpoint),
            "adapter_sha256": sha256(ckpt),
            "role": "diagnostic only, no fit, fixed zero yaw and 30km/h desired speed",
            "no_visual_input": True,
            "counterfactual": "adapter load channels only",
            "brake_receipts_sha256": sha256(audit / "moving-brake-receipts.json"),
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / "outputs.npz", times=times, outputs=outputs, braking=selection)
    out = {
        "status": "completed_pace_diagnostic_brake_load_graph_replay",
        "fixed_speed_task_not_original_pace_task": True,
        "frames": len(rows),
        "moving_braking_frames": int(selection.sum()),
        "max_interval_seconds": float(np.diff(times).max()),
        "output_columns": ["steering", "gas", "brake"],
        "braking_mean_abs_output_delta": np.abs(delta[selection]).mean(axis=0).tolist(),
        "braking_max_abs_output_delta": np.abs(delta[selection]).max(axis=0).tolist(),
        "outputs_sha256": sha256(run / "outputs.npz"),
        "optimizer_updates": 0,
        "no_visual_input": True,
        "host_receipt_not_physics_time": True,
        "adapter_only_intervention": True,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
