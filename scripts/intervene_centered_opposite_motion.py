"""Counterfactual TRAIN-mean-centered T4/T5 channel reversal on saved histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training = Path(args.training)
    receipt = read(training / "result.json")
    manifest = read(training / "manifest.json")
    plan = manifest["config"]["plan"]
    if (
        receipt["status"] != "completed_fixed_joint_command_training"
        or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Original fixed joint training checkpoint required")
    source = read(Path(plan["source"]) / "result.json")
    if sha256(source["checkpoint"]) != source["checkpoint_sha256"]:
        raise ValueError("Original source checkpoint changed")
    if not torch.cuda.is_available():
        raise ValueError("GPU required for original cached graph")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    checkpoint = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["spec"] != model.spec or checkpoint["schema"] != model.schema:
        raise ValueError("Original graph/spec mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    candidate = Path(args.candidate)
    trained = read(candidate / "result.json")
    if (
        trained["status"] != "completed_one_pass_hsvs_boundary_train"
        or trained["source_checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != trained["checkpoint_sha256"]
    ):
        raise ValueError("Exact completed HS/VS boundary candidate required")
    boundary = torch.load(candidate / "visual-boundary.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        model.backbone.vision.weight.copy_(boundary["weight"].to("cuda"))
        model.backbone.vision.bias.copy_(boundary["bias"].to("cuda"))
    spec = read(plan["source_spec"])
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        types = graph["types"].copy()
    comparison = read(Path(args.comparison) / "result.json")
    if (
        comparison["status"] != "completed_four_view_isolated_motion_interventions"
        or sha256(Path(args.comparison) / "predictions.npz") != comparison["predictions_sha256"]
        or sha256(spec["graph_file"]) != spec["graph_sha256"]
    ):
        raise ValueError("Pinned graph and previous prediction evidence required")
    positions = torch.tensor(np.flatnonzero(types == "DNb06"), device="cuda")
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        if list(graph["ids"][positions.cpu().numpy()]) != ["10888", "11067"]:
            raise ValueError("Exact DNb06 left/right identities required")
    prior_root = Path(args.prior)
    old_receipt = read(prior_root / "result.json")
    if (
        old_receipt["status"] != "completed_dnb06_original_history_state_audit"
        or sha256(prior_root / "predictions.npz") != old_receipt["predictions_sha256"]
    ):
        raise ValueError("Unchanged prior natural history audit required")
    root = Path(plan["data"])
    dataset = read(root / "dataset.json")
    entries = dataset["episodes"]
    if (
        sum(e["role"] == "train" for e in entries) != 6
        or sum(e["role"] == "development" for e in entries) != 2
    ):
        raise ValueError("Original six TRAIN and two known DEVELOPMENT histories required")
    conditions = ("intact", "opposite_motion_channels")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "training_result_sha256": sha256(training / "result.json"),
            "candidate_checkpoint_sha256": trained["checkpoint_sha256"],
            "dataset_sha256": sha256(root / "dataset.json"),
            "prior_predictions_sha256": old_receipt["predictions_sha256"],
            "intervention": ("swap paired motion deviations from TRAIN mean in all views; "
                "posthoc control"),
            "direct_ports": "all motion features TRAIN mean in both conditions",
            "known_development_not_blind": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    direct_ports = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    reports, saved = [], {}
    with torch.no_grad():
        for entry in entries:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original episode changed")
            with load_array(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
            t = {k: torch.tensor(arrays[k], device="cuda") for k in names}
            if t["visual_valid"].any():
                raise ValueError("Original visual mask must be preserved")
            original = t["scene_features"].reshape(-1, 4, 9, 16)
            baseline = model.backbone.scene_command.mean.reshape(1, 4, 9, 16)
            deviation = original - baseline
            opposite = original.clone()
            opposite[:, :, 1:] = baseline[:, :, 1:] + deviation[:, :, [2, 1, 4, 3, 6, 5, 8, 7]]
            outside_count = int(((opposite[:, :, 1:] < -1) | (opposite[:, :, 1:] > 1)).sum())
            opposite = opposite.flatten(1)
            mean_scene = t["scene_features"].clone()
            select = torch.zeros(576, dtype=torch.bool, device="cuda").reshape(4, 9, 16)
            select[:, 1:] = True
            select = select.flatten()
            mean_scene[:, select] = model.backbone.scene_command.mean[select]
            drives = [
                model._drive_tensor(*(t[k] for k in names[:5]), x)
                for x in (t["scene_features"], opposite, mean_scene)
            ]
            for drive in drives[:2]:
                drive[:, direct_ports] = drives[2][:, direct_ports]
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays["observation_time"])]
            states = [model.initial_state(1), model.initial_state(1)]
            predictions, neurons = ([[] for _ in conditions] for _ in range(2))
            held = None
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                        states = [
                            model._advance_elapsed_tensor(held[k], states[k], dt) for k in range(2)
                        ]
                for k, state in enumerate(states):
                    predictions[k].append(model._read_state_tensor(state)[0][0])
                    neurons[k].append(state[0, positions])
                held = [drive[index : index + 1] for drive in drives[:2]]
            outputs = [torch.stack(values).cpu().numpy() for values in predictions]
            neural = [torch.stack(values).cpu().numpy() for values in neurons]
            prefix = entry["file"].removesuffix(".npz")
            with load_array(prior_root / "predictions.npz", allow_pickle=False) as old:
                np.testing.assert_allclose(
                    outputs[0], old[prefix + "_intact_intent"], rtol=0, atol=1e-6
                )
                np.testing.assert_allclose(
                    neural[0], old[prefix + "_intact_state"], rtol=0, atol=1e-6
                )
                mean_output = old[prefix + "_motion_mean_intent"].copy()
            valid = arrays["target_valid"][:, 0]
            if not valid.any():
                raise ValueError("Original valid steering labels required")
            target = arrays["target"][valid, 0]
            metrics = {}
            for name, output in zip(conditions, outputs, strict=True):
                saved[prefix + "_" + name + "_intent"] = output
                saved[prefix + "_" + name + "_state"] = neural[conditions.index(name)]
                metrics[name] = {
                    "steering_rmse": float(np.sqrt(np.mean((output[valid, 0] - target) ** 2)))
                }
            delta = outputs[0][:, 0] - outputs[1][:, 0]
            vs_mean = outputs[0][:, 0] - mean_output[:, 0]
            metrics["opposite_intervention"] = {
                "steering_mean_abs_change": float(np.abs(delta).mean()),
                "steering_max_abs_change": float(np.abs(delta).max()),
                "original_vs_mean_steering_mean_abs": float(np.abs(vs_mean).mean()),
                "outside_tanh_feature_bounds_count": outside_count,
            }
            reports.append(
                {
                    "file": entry["file"],
                    "role": entry["role"],
                    "observations": len(schedule),
                    "metrics": metrics,
                }
            )
            print(json.dumps(reports[-1]), flush=True)
    np.savez_compressed(run / "predictions.npz", **saved)
    if (
        sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != trained["checkpoint_sha256"]
    ):
        raise ValueError("Fixed checkpoint changed")
    write_json(
        run / "result.json",
        {
            "status": "completed_centered_opposite_motion_channel_replay",
            "reports": reports,
            "predictions_sha256": sha256(run / "predictions.npz"),
            "counterfactual_not_image_resimulation": True,
            "known_development_not_independent": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )


if __name__ == "__main__":
    main()
