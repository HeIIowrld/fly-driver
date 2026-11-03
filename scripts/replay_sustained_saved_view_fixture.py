"""Replay plus/minus slow-path fixtures on original known development histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_sustained_opponent_boundary import (
    SustainedOpponentBoundary,
    opponent_axes,
)
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "prior", "output"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    training, candidate, prior = map(Path, (a.training, a.candidate, a.prior))
    tr, cr, pr = (read(p / "result.json") for p in (training, candidate, prior))
    plan = read(training / "manifest.json")["config"]["plan"]
    root = Path(plan["data"])
    source = read(Path(plan["source"]) / "result.json")
    config = read(prior / "manifest.json")["config"]
    if (sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
            or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
            or sha256(candidate / "temporal-boundary.pt") != cr["checkpoint_sha256"]
            or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or cr["dataset_sha256"] != sha256(root / "dataset.json")
            or sha256(prior / "predictions.npz") != pr["predictions_sha256"]
            or config["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or config["candidate_checkpoint_sha256"] != cr["checkpoint_sha256"]
            or not torch.cuda.is_available()):
        raise ValueError("Exact model, candidate, dataset and prior replay required")
    entries = [e for e in read(root / "dataset.json")["episodes"] if e["role"] == "development"]
    if len(entries) != 2:
        raise ValueError("Two original known development histories required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan["source_spec"], torch.load(
        source["checkpoint"], map_location="cpu", weights_only=True)).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Original v7 model contract required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    saved = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    boundary = SustainedOpponentBoundary(model.backbone.scene_command.mean,
        model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to("cuda")
    missing, unexpected = boundary.load_state_dict(saved["model"], strict=False)
    if missing != ["sustained_projection.weight"] or unexpected or saved["actor_qualified"]:
        raise ValueError("Only new unqualified sustained projection may be missing")
    boundary.eval().requires_grad_(False)
    run = begin_run(a.output, {"program_sha256": sha256(__file__),
        "boundary_sha256": sha256("src/flybrain_drive/flight_sustained_opponent_boundary.py"),
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "temporal_checkpoint_sha256": cr["checkpoint_sha256"],
        "dataset_sha256": cr["dataset_sha256"],
        "prior_predictions_sha256": pr["predictions_sha256"],
        "conditions": ["original", "transient", "fixture_plus", "fixture_minus"],
        "fixture": "slow weights = +/- original trained transient weights; no new learning",
        "direct_motion": "original complete motion path", "known_development_not_blind": True,
        "labels_opened_after_inference": True, "actor_deployed": False, "optimizer_updates": 0})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
    reports, arrays = [], {}
    with torch.no_grad():
        weight = boundary.opponent_projection.weight.detach().clone()
        for entry in entries:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original saved history changed")
            with load_array(path, allow_pickle=False) as data:
                times = data["observation_time"].copy()
                t = {key: torch.tensor(data[key], device="cuda") for key in names}
            complete_history(times, times)
            if t["visual_valid"].any():
                raise ValueError("Original visual6 mask must remain invalid")
            base = model._drive_tensor(*(t[k] for k in names))
            state, traces, held, outputs = model.initial_state(4), None, None, []
            views = torch.ones((1, 4), dtype=torch.bool, device="cuda")
            for i in range(len(times)):
                elapsed = None if i == 0 else float(times[i] - times[i - 1])
                if i:
                    for dt in substeps(elapsed):
                        state = model._advance_elapsed_tensor(held, state, dt)
                outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
                transient, traces = boundary.step(
                    t["scene_features"][i:i+1], views, traces, elapsed
                )
                slow = opponent_axes(traces[1])
                fast = opponent_axes(traces[0] - traces[1])
                logits = boundary.opponent_projection(fast)
                plus = .5 * torch.tanh(logits + slow @ weight.T)
                minus = .5 * torch.tanh(logits - slow @ weight.T)
                b = base[i:i+1]
                held = torch.cat([b] + [boundary.modulate(b, model.backbone.vision_ports, x)
                                        for x in (transient, plus, minus)])
            output = np.stack(outputs)
            stem = entry["file"].removesuffix(".npz")
            with load_array(prior / "predictions.npz", allow_pickle=False) as old:
                reference = old[stem + "_output"].copy()
            error = float(np.max(np.abs(output[:, :2] - reference[:, :2])))
            if error > 1e-6:
                raise ValueError(f"Original and transient conditions differ from prior: {error}")
            with load_array(path, allow_pickle=False) as data:
                target, valid = data["target"].copy(), data["target_valid"].copy()
            heads = {}
            for j, name in enumerate(("steering", "forward")):
                mask = valid[:, j]
                if not mask.any():
                    raise ValueError("Valid target required")
                pred = output[mask, :, j]
                heads[name] = {"valid": int(mask.sum()),
                    "rmse": np.sqrt(np.mean((pred - target[mask, j, None])**2, axis=0)).tolist(),
                    "zero_rmse": float(np.sqrt(np.mean(target[mask, j]**2))),
                    "mean_abs_fixture_change": [float(np.mean(np.abs(pred[:, k] - pred[:, 1])))
                                                for k in (2, 3)]}
            reports.append({"file": entry["file"], "observations": len(times),
                            "previous_replay_max_error": error, "heads": heads})
            arrays[stem + "_time"], arrays[stem + "_output"] = times, output
            print(entry["file"], heads, flush=True)
    np.savez_compressed(run / "predictions.npz", **arrays)
    write_json(run / "result.json", {"status": "completed_sustained_saved_view_fixture_replay",
        "reports": reports, "predictions_sha256": sha256(run / "predictions.npz"),
        "actor_deployed": False, "optimizer_updates": 0})


if __name__ == "__main__":
    main()
