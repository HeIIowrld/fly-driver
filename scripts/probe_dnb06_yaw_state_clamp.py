"""Clamp DNb06 to same-stimulus zero-yaw states in a fixed synthetic model experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training, candidate = Path(args.training), Path(args.candidate)
    receipt, plan = (
        read(training / "result.json"),
        read(training / "manifest.json")["config"]["plan"],
    )
    selected = read(candidate / "result.json")
    if (
        receipt["status"] != "completed_fixed_joint_command_training"
        or selected["status"] != "completed_one_pass_hsvs_boundary_train"
        or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != selected["checkpoint_sha256"]
        or selected["source_checkpoint_sha256"] != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Exact frozen graph and HS/VS candidate required")
    source = read(Path(plan["source"]) / "result.json")
    if sha256(source["checkpoint"]) != source["checkpoint_sha256"] or not torch.cuda.is_available():
        raise ValueError("Original source graph and GPU required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    boundary = torch.load(candidate / "visual-boundary.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        model.backbone.vision.weight.copy_(boundary["weight"].to("cuda"))
        model.backbone.vision.bias.copy_(boundary["bias"].to("cuda"))
    spec = read(plan["source_spec"])
    if sha256(spec["graph_file"]) != spec["graph_sha256"]:
        raise ValueError("Original graph changed")
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        types = graph["types"].copy()
    positions = torch.tensor(np.flatnonzero(types == "DNb06"), device="cuda")
    if len(positions) != 2:
        raise ValueError("Exact original DNb06 pair required")
    root = Path(plan["data"])
    dataset = read(root / "dataset.json")
    entry = next(e for e in dataset["episodes"] if e["role"] == "train")
    episode = root / entry["file"]
    if sha256(episode) != entry["sha256"]:
        raise ValueError("Original context episode changed")
    with load_array(episode, allow_pickle=False) as data:
        values = {
            k: torch.tensor(data[k][0:1], device="cuda")
            for k in ("visual", "visual_valid", "body", "body_valid", "goals")
        }
    conditions = [
        {"yaw_rad_s": yaw, "sign": sign} for yaw in (-0.5, 0.0, 0.5) for sign in (0, -1, 1)
    ]
    count = len(conditions)
    scene = model.backbone.scene_command.mean[None].repeat(count, 1)
    shaped = scene.reshape(count, 4, 9, 16)
    scale = model.backbone.scene_command.scale.reshape(4, 9, 16)
    for index, condition in enumerate(conditions):
        # Left-view T5c: an explicitly synthetic feature pulse, not a moving image.
        shaped[index, 1, 7, :] += condition["sign"] * scale[1, 7, :]
    repeated = {key: value.repeat(count, 1) for key, value in values.items()}
    if not repeated["body_valid"][:, 4].all():
        raise ValueError("Original yaw channel must be valid")
    repeated["body"][:, 4] = torch.tensor([c["yaw_rad_s"] for c in conditions], device="cuda")
    with torch.no_grad():
        active_drive = model._drive_tensor(*(repeated[k] for k in values), scene)
        baseline_scene = model.backbone.scene_command.mean[None].repeat(count, 1)
        baseline_drive = model._drive_tensor(*(repeated[k] for k in values), baseline_scene)
        direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
        active_drive[:, direct] = baseline_drive[:, direct]
        if not torch.equal(active_drive[:, direct], baseline_drive[:, direct]):
            raise ValueError("Direct motion bypass must be held to baseline")
        run = begin_run(
            args.output,
            {
                "program_sha256": sha256(__file__),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "visual_candidate_sha256": selected["checkpoint_sha256"],
                "graph_sha256": spec["graph_sha256"],
                "episode_sha256": sha256(episode),
                "stimulus": "left T5c synthetic pooled features mean +/- 1 TRAIN std, 16 bins",
                "body_yaw_rad_s": [-0.5, 0.0, 0.5],
                "schedule_seconds": [0.5, 0.25, 0.75],
                "dt_seconds": 0.01,
                "direct_motion_ports_clamped": True,
                "reference_result_sha256": sha256(Path(args.reference) / "result.json"),
                "intervention": "DNb06 states fixed to equal-visual-stimulus zero-yaw trajectory",
                "optimizer_updates": 0,
            },
        )
        (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
        active_drive = active_drive.repeat(2, 1)
        baseline_drive = baseline_drive.repeat(2, 1)
        state = model.initial_state(count * 2)
        references = torch.tensor([3 + i % 3 for i in range(count)], device="cuda")
        traces, output = [], []
        for step in range(150):
            drive = active_drive if 50 <= step < 75 else baseline_drive
            state = model._advance_elapsed_tensor(drive, state, 0.01)
            state[count:, positions] = state[references][:, positions]
            traces.append(state[:, positions].detach().cpu().numpy())
            output.append(model._read_state_tensor(state)[0].detach().cpu().numpy())
            if (step + 1) % 50 == 0:
                print(json.dumps({"completed_ticks": step + 1}), flush=True)
    traces, output = np.stack(traces), np.stack(output)
    if not np.isfinite(traces).all() or not np.isfinite(output).all():
        raise ValueError("Finite original circuit responses required")
    reference = Path(args.reference)
    reference_result = read(reference / "result.json")
    if (
        reference_result["status"] != "completed_synthetic_visual_body_yaw_interaction"
        or sha256(reference / "responses.npz") != reference_result["responses_sha256"]
    ):
        raise ValueError("Original synthetic trial and verified response required")
    with np.load(reference / "responses.npz", allow_pickle=False) as prior:
        state_error = float(np.max(np.abs(traces[:, :count] - prior["dnb06"])))
        output_error = float(np.max(np.abs(output[:, :count] - prior["intent"])))
    if max(state_error, output_error) > 1e-6:
        raise ValueError("Original unclamped replay differs")
    if not np.array_equal(traces[:, count:], traces[:, [3 + i % 3 for i in range(count)]]):
        raise ValueError("DNb06 clamp differs from zero-yaw reference")
    np.savez_compressed(run / "responses.npz", dnb06=traces, intent=output)
    reports = []
    for index, condition in enumerate(conditions):
        clamp_delta = output[:, index + count] - output[:, index]
        reports.append(
            {
                **condition,
                "baseline_end_intent_change": clamp_delta[49].tolist(),
                "pulse_end_intent_change": clamp_delta[74].tolist(),
                "peak_abs_steering_change": float(np.max(np.abs(clamp_delta[:, 0]))),
            }
        )
    if (
        sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != selected["checkpoint_sha256"]
        or sha256(episode) != entry["sha256"]
    ):
        raise ValueError("Fixed evidence changed during experiment")
    result = {
        "status": "completed_dnb06_synthetic_yaw_state_clamp",
        "conditions": reports,
        "responses_sha256": sha256(run / "responses.npz"),
        "stimulus_is_not_image_or_measured_fly_response": True,
        "normalization_scale_is_train_only": True,
        "direct_ports_fixed": True,
        "original_state_max_abs_error": state_error,
        "original_intent_max_abs_error": output_error,
        "clamp_matches_same_visual_zero_yaw": True,
        "closed_loop_or_physiology_verified": False,
        "actor_deployed": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps({"status": result["status"], "conditions": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
