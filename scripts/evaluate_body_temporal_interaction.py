"""Replay transferred body and temporal boundaries on known development histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "temporal", "body", "prior", "output"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    training, temporal_dir, body_dir, prior = map(Path, (a.training, a.temporal, a.body, a.prior))
    tr, tc, bc, previous = [
        read(p / "result.json") for p in (training, temporal_dir, body_dir, prior)
    ]
    plan = read(training / "manifest.json")["config"]["plan"]
    root = Path(plan["data"])
    source = read(Path(plan["source"]) / "result.json")
    if (sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
            or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
            or sha256(temporal_dir / "temporal-boundary.pt") != tc["checkpoint_sha256"]
            or sha256(body_dir / "body-boundary.pt") != bc["checkpoint_sha256"]
            or tc["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or bc["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or tc["dataset_sha256"] != sha256(root / "dataset.json")
            or sha256(prior / "predictions.npz") != previous["predictions_sha256"]
            or not torch.cuda.is_available()):
        raise ValueError("Source, candidates, dataset and prior must match")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan["source_spec"], torch.load(
        source["checkpoint"], map_location="cpu", weights_only=True)).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Original model contract required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    tb = torch.load(temporal_dir / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    bb = torch.load(body_dir / "body-boundary.pt", map_location="cpu", weights_only=True)
    if (tb["actor_qualified"] is not False or bb["actor_qualified"] is not False
            or bb["centered_availability"] is not True
            or bb["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or tb["source_checkpoint_sha256"] != tr["checkpoint_sha256"]):
        raise ValueError("Research-only candidate provenance required")
    temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
        model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to("cuda")
    temporal.load_state_dict(tb["model"], strict=True)
    temporal.eval().requires_grad_(False)
    joint = ExperimentalVisualBodyAirflow(model, centered_availability=True).to("cuda").eval()
    run = begin_run(a.output, {"program_sha256": sha256(__file__),
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "temporal_checkpoint_sha256": tc["checkpoint_sha256"],
        "body_checkpoint_sha256": bc["checkpoint_sha256"],
        "body_training_visual_candidate_sha256": bb["visual_candidate_sha256"],
        "prior_predictions_sha256": previous["predictions_sha256"],
        "dataset_sha256": sha256(root / "dataset.json"),
        "conditions": ["baseline", "body_transfer", "temporal", "both"],
        "airflow": "missing, all axes invalid", "known_development_not_blind": True,
        "body_transfer": "trained with another visual boundary and mean-clamped direct motion",
        "teacher_labels_opened_after_inference": True, "optimizer_updates": 0,
        "actor_deployed": False})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    entries = [e for e in read(root / "dataset.json")["episodes"] if e["role"] == "development"]
    if len(entries) != 2:
        raise ValueError("Two known development histories required")
    names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
    reports, saved = [], {}
    with torch.no_grad():
        for entry in entries:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("History hash differs")
            with load_array(path, allow_pickle=False) as data:
                times = data["observation_time"].copy()
                t = {key: torch.tensor(data[key], device="cuda") for key in names}
            complete_history(times, times)
            if t["visual_valid"].any():
                raise ValueError("Original visual6 mask must remain invalid")
            count = len(times)
            views = torch.ones((count, 4), dtype=torch.bool, device="cuda")
            wind = torch.zeros((count, 3), device="cuda")
            wind_valid = torch.zeros((count, 3), dtype=torch.bool, device="cuda")
            joint.body_feedback.projection.weight.zero_()
            baseline = joint.drive(*(t[k] for k in names), views, wind, wind_valid)
            joint.body_feedback.projection.weight.copy_(bb["weight"].to("cuda"))
            body = joint.drive(*(t[k] for k in names), views, wind, wind_valid)
            state, trace, held, outputs = model.initial_state(4), None, None, []
            for i in range(count):
                elapsed = None if i == 0 else float(times[i] - times[i - 1])
                if i:
                    for dt in substeps(elapsed):
                        state = model._advance_elapsed_tensor(held, state, dt)
                outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
                delta, trace = temporal.step(
                    t["scene_features"][i:i+1], views[i:i+1], trace, elapsed
                )
                b, d = baseline[i:i+1], body[i:i+1]
                held = torch.cat((b, d, temporal.modulate(b, model.backbone.vision_ports, delta),
                                  temporal.modulate(d, model.backbone.vision_ports, delta)))
            output = np.stack(outputs)
            stem = entry["file"].removesuffix(".npz")
            with load_array(prior / "predictions.npz", allow_pickle=False) as old:
                reference = old[stem + "_output"].copy()
            error = float(np.max(np.abs(output[:, [0, 2]] - reference[:, [0, 1]])))
            if error > 1e-6:
                raise ValueError(f"Previous baseline or temporal replay mismatch: {error}")
            with load_array(path, allow_pickle=False) as data:
                truth, valid = data["target"].copy(), data["target_valid"].copy()
            metrics = {}
            for j, name in enumerate(("steering", "forward")):
                mask = valid[:, j]
                if not mask.any():
                    raise ValueError("Empty target mask")
                y = output[mask, :, j]
                metrics[name] = {"valid": int(mask.sum()),
                    "rmse": np.sqrt(np.mean((y - truth[mask, j, None])**2, axis=0)).tolist(),
                    "zero_rmse": float(np.sqrt(np.mean(truth[mask, j]**2))),
                    "mean_abs_body_effect": float(np.mean(np.abs(y[:, 1] - y[:, 0]))),
                    "mean_abs_temporal_effect": float(np.mean(np.abs(y[:, 2] - y[:, 0]))),
                    "mean_abs_interaction": float(np.mean(np.abs(
                        y[:, 3] - y[:, 2] - y[:, 1] + y[:, 0])))}
            reports.append({"file": entry["file"], "observations": count,
                            "previous_replay_max_error": error, "heads": metrics})
            saved[stem + "_time"], saved[stem + "_output"] = times, output
            print(entry["file"], metrics, flush=True)
    np.savez_compressed(run / "predictions.npz", **saved)
    write_json(run / "result.json", {"status": "completed_body_temporal_interaction_replay",
        "reports": reports, "predictions_sha256": sha256(run / "predictions.npz"),
        "actor_deployed": False, "optimizer_updates": 0})


if __name__ == "__main__":
    main()
