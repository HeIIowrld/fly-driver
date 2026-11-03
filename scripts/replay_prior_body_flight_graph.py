"""Compare paired original/prior body streams in the frozen flight graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_body_feedback_adapter import VehicleBodyFeedbackAdapter
from flybrain_drive.flight_embodied_navigation import modulate_current
from flybrain_drive.graph import sha256
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("body-pairs", "candidate", "fit-plan", "fit-result",
                 "pedal-checkpoint", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    pairs_root, candidate = Path(args.body_pairs), Path(args.candidate)
    proof = read(pairs_root / "result.json")
    samples_path = pairs_root / "body-pairs.npz"
    if (proof["status"] != "completed_prior_body_counterfactual"
            or proof["training_admissible"] is not False
            or sha256(samples_path) != proof["body_pairs_sha256"]):
        raise ValueError("Exact fixed prior/body comparison required")
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
    with np.load(samples_path, allow_pickle=False) as data:
        pairs = data["values"].copy().astype(np.float32)
        masks = data["valid"].copy()
        times = data["decision_monotonic_seconds"].copy()
    if (pairs.ndim != 3 or pairs.shape[1:] != (2, 21) or masks.shape != pairs.shape
            or len(times) != len(pairs) or len(times) < 2 or not np.isfinite(pairs).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .08)):
        raise ValueError("Finite complete original ordered body pairs required")
    body, changed = pairs[:, 0], pairs[:, 1]
    valid, changed_valid = masks[:, 0], masks[:, 1]
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
    alternate_valid = torch.tensor(changed_valid, device="cuda")
    command = torch.tensor([[0.0, 30.0 / 3.6]], device="cuda").expand(len(times), -1)
    goals = torch.tensor([[0.0, 1.0]], device="cuda").expand(len(times), -1)
    with torch.no_grad():
        fixed_drive = base.model.drive(command, b, v, goals, **base.kwargs)
        ports = base.model.backbone.body_ports
        actual = fixed_drive.clone()
        counterfactual = base.model.drive(command, alternate, alternate_valid, goals, **base.kwargs)
        actual[:, ports] = modulate_current(fixed_drive[:, ports], adapter(b, v))
        counterfactual[:, ports] = modulate_current(
            counterfactual[:, ports], adapter(alternate, alternate_valid))
        state = quiet.expand(2, -1).clone()
        outputs = []
        for index in range(len(times)):
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
    if not np.isfinite(outputs).all():
        raise ValueError("Finite outputs required")
    delta = outputs[:, 0] - outputs[:, 1]
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "native_samples_sha256": sha256(samples_path),
            "source_audit_sha256": sha256(pairs_root / "result.json"),
            "fit_plan_sha256": sha256(args.fit_plan),
            "fit_result_sha256": sha256(args.fit_result),
            "pedal_checkpoint_sha256": sha256(args.pedal_checkpoint),
            "adapter_sha256": sha256(ckpt),
            "role": "diagnostic only, no fit, fixed zero yaw and 30km/h desired speed",
            "no_visual_input": True,
            "counterfactual": "all body channels and masks, original versus prior receipt",
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / "outputs.npz", times=times, outputs=outputs)
    out = {
        "status": "completed_prior_body_flight_graph_replay",
        "fixed_speed_task_not_original_pace_task": True,
        "frames": len(times),
        "max_interval_seconds": float(np.diff(times).max()),
        "output_columns": ["steering", "gas", "brake"],
        "mean_abs_output_delta": np.abs(delta).mean(axis=0).tolist(),
        "max_abs_output_delta": np.abs(delta).max(axis=0).tolist(),
        "outputs_sha256": sha256(run / "outputs.npz"),
        "optimizer_updates": 0,
        "no_visual_input": True,
        "host_receipt_not_physics_time": True,
        "all_body_inputs_intervened": True,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
