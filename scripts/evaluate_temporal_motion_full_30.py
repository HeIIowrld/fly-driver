"""Evaluate full-motion temporal candidate on same-goal development histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_temporal_motion_boundary import TemporalMotionBoundary
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "prior", "original-prior", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    training, candidate, prior = map(Path, (a.training, a.candidate, a.prior))
    tr, tm, cr, pr = (
        read(training / "result.json"),
        read(training / "manifest.json"),
        read(candidate / "result.json"),
        read(prior / "result.json"),
    )
    plan, pm = tm["config"]["plan"], read(prior / "manifest.json")["config"]
    root = Path(plan["data"])
    dataset = read(root / "dataset.json")
    source = read(Path(plan["source"]) / "result.json")
    original_prior = Path(a.original_prior)
    original_result = read(original_prior / "result.json")
    original_config = read(original_prior / "manifest.json")["config"]
    original_predictions = original_prior / "development-predictions.npz"
    module = "src/flybrain_drive/flight_temporal_motion_boundary.py"
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or cr["status"] != "completed_temporal_motion_boundary_onepass"
        or cr["program_sha256"] != sha256("scripts/train_temporal_motion_full_30.py")
        or cr["direct_motion_held_mean"] is not False
        or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or cr["boundary_source_sha256"] != sha256(module)
        or cr["dataset_sha256"] != sha256(root / "dataset.json")
        or sha256(candidate / "temporal-boundary.pt") != cr["checkpoint_sha256"]
        or pr["status"] != "completed_direct_descending_clamp_replay"
        or sha256(prior / "group-differences.npz") != pr["group_difference_sha256"]
        or pm["checkpoint_sha256"] != tr["checkpoint_sha256"]
        or original_result["status"] != "completed_fixed_embodied_development_evaluation"
        or original_config["checkpoint_sha256"] != tr["checkpoint_sha256"]
        or original_config["training_result_sha256"] != sha256(training / "result.json")
        or not torch.cuda.is_available()
    ):
        raise ValueError("Exact original, qualified trained candidate and fixed baseline required")
    entries = [e for e in dataset["episodes"] if e["role"] == "development"]
    original_hash = sha256(original_predictions)
    if len(entries) != 2:
        raise ValueError("Exactly two original known development episodes required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Original fixed graph contract required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    saved = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    if (
        saved["schema"] != "flybrain.temporal_motion_boundary.v1"
        or saved["actor_qualified"] is not False
        or saved["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or saved["time_constants_seconds"] != [0.05, 0.2]
    ):
        raise ValueError("Exact fixed temporal candidate and declared time constants required")
    temporal = TemporalMotionBoundary(
        model.backbone.scene_command.mean,
        model.backbone.scene_command.scale,
        len(model.backbone.vision_ports),
    ).to("cuda")
    temporal.load_state_dict(saved["model"], strict=True)
    temporal.eval().requires_grad_(False)
    if not torch.equal(temporal.mean, model.backbone.scene_command.mean) or not torch.equal(
        temporal.scale, model.backbone.scene_command.scale
    ):
        raise ValueError("Original TRAIN-only temporal normalization required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "source_checkpoint_sha256": tr["checkpoint_sha256"],
            "candidate_checkpoint_sha256": cr["checkpoint_sha256"],
            "prior_group_difference_sha256": pr["group_difference_sha256"],
            "prior_original_predictions_sha256": original_hash,
            "prior_original_hash_previously_pinned": False,
            "dataset_sha256": sha256(root / "dataset.json"),
            "conditions": [
                "zero_temporal_boundary", "learned_temporal_boundary", "mean_trace",
                "clamped_reference",
            ],
            "motion_traces_mean_intervention": "TRAIN mean input yields zero temporal delta",
            "direct_motion": "original full four-view motion, separate clamped reference",
            "teacher_labels_opened_after_inference": True,
            "known_development_not_blind": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
    direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    reports, arrays = [], {}
    with torch.no_grad():
        for entry in entries:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original role-bound history changed")
            with load_array(path, allow_pickle=False) as data:
                times = data["observation_time"].copy()
                t = {key: torch.tensor(data[key], device="cuda") for key in names}
            complete_history(times, times)
            if t["visual_valid"].any():
                raise ValueError("Original visual6 availability must remain unchanged")
            base = model._drive_tensor(*(t[key] for key in names))
            mean_scene = t["scene_features"].clone()
            motion = torch.zeros((4, 9, 16), dtype=torch.bool, device="cuda")
            motion[:, 1:] = True
            motion = motion.flatten()
            mean_scene[:, motion] = model.backbone.scene_command.mean[motion]
            mean_drive = model._drive_tensor(*(t[key] for key in names[:5]), mean_scene)
            clamped = base.clone()
            clamped[:, direct] = mean_drive[:, direct]
            state, traces, held, predictions, trace_norm = (
                model.initial_state(4),
                None,
                None,
                [],
                [],
            )
            mean_traces = None
            for index in range(len(times)):
                elapsed = None if index == 0 else float(times[index] - times[index - 1])
                if index:
                    for dt in substeps(elapsed):
                        state = model._advance_elapsed_tensor(held, state, dt)
                predictions.append(model._read_state_tensor(state)[0].cpu().numpy())
                delta, traces = temporal.step(
                    t["scene_features"][index : index + 1],
                    torch.ones((1, 4), dtype=torch.bool, device="cuda"),
                    traces,
                    elapsed,
                )
                trace_norm.append(float((traces[0] - traces[1]).norm()))
                learned = temporal.modulate(
                    base[index : index + 1], model.backbone.vision_ports, delta
                )
                mean_delta, mean_traces = temporal.step(
                    mean_scene[index : index + 1],
                    torch.ones((1, 4), dtype=torch.bool, device="cuda"),
                    mean_traces,
                    elapsed,
                )
                mean_trace_drive = temporal.modulate(
                    base[index : index + 1], model.backbone.vision_ports, mean_delta
                )
                held = torch.cat([base[index : index + 1], learned, mean_trace_drive,
                                  clamped[index : index + 1]])
            output = np.stack(predictions)
            stem = entry["file"].removesuffix(".npz")
            with load_array(prior / "group-differences.npz", allow_pickle=False) as old:
                difference = old[stem + "_clamp_intent"].copy()
            with load_array(original_predictions, allow_pickle=False) as old:
                prior_out = old[stem + "_intact"].copy()
            baseline_error = float(np.max(np.abs(output[:, 0] - prior_out)))
            clamp_error = float(np.max(np.abs(output[:, 3] - (prior_out - difference))))
            if clamp_error > 1e-6:
                raise ValueError("Original direct-clamped reference mismatch")
            if baseline_error > 1e-6:
                raise ValueError(
                    "Zero temporal baseline must reproduce earlier full-motion model"
                )
            mean_trace_error = float(np.max(np.abs(output[:, 0] - output[:, 2])))
            if mean_trace_error > 1e-6:
                raise ValueError("Constant mean motion must reproduce zero temporal baseline")
            with load_array(path, allow_pickle=False) as data:
                truth, valid = data["target"].copy(), data["target_valid"].copy()
            if truth.shape != (len(times), 2) or valid.shape != truth.shape or valid.dtype != bool:
                raise ValueError("Original two-head masked target required")
            metrics = {}
            for i, name in enumerate(("steering", "forward")):
                mask = valid[:, i]
                if not mask.any():
                    raise ValueError("Original nonempty target masks required")
                metrics[name] = {
                    "valid": int(mask.sum()),
                    "baseline_rmse": float(
                        np.sqrt(np.mean((output[mask, 0, i] - truth[mask, i]) ** 2))
                    ),
                    "candidate_rmse": float(
                        np.sqrt(np.mean((output[mask, 1, i] - truth[mask, i]) ** 2))
                    ),
                    "zero_rmse": float(np.sqrt(np.mean(truth[mask, i] ** 2))),
                    "mean_abs_candidate_change": float(
                        np.mean(np.abs(output[mask, 1, i] - output[mask, 0, i]))
                    ),
                }
            reports.append(
                {
                    "file": entry["file"],
                    "observations": len(times),
                    "baseline_error": baseline_error,
                    "clamped_reference_error": clamp_error,
                    "mean_trace_ablation_error": mean_trace_error,
                    "max_trace_norm": max(trace_norm),
                    "heads": metrics,
                }
            )
            arrays[stem + "_time"], arrays[stem + "_output"] = times, output
            arrays[stem + "_target"], arrays[stem + "_valid"] = truth, valid
            arrays[stem + "_trace_norm"] = np.asarray(trace_norm)
            print(json.dumps(reports[-1]), flush=True)
    if sha256(original_predictions) != original_hash:
        raise ValueError("Prior original predictions changed during evaluation")
    np.savez_compressed(run / "predictions.npz", **arrays)
    result = {
        "status": "completed_temporal_motion_known_development_evaluation",
        "reports": reports,
        "predictions_sha256": sha256(run / "predictions.npz"),
        "known_development_not_independent": True,
        "direct_dn_motion_held_mean": False,
        "physiology_or_driving_verified": False,
        "actor_deployed": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
