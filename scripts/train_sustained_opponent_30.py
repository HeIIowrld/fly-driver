"""Fit sustained opponent pathway on original TRAIN histories; no deployment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_sustained_opponent_boundary import SustainedOpponentBoundary
from flybrain_drive.graph import sha256
from scripts.qualify_flight_pedal_graph_v2 import tensor_digest
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=("qualify", "train"), required=True)
    p.add_argument("--qualification")
    a = p.parse_args()
    training = Path(a.training)
    tr, tm = read(training / "result.json"), read(training / "manifest.json")
    plan, module = (
        tm["config"]["plan"],
        Path("src/flybrain_drive/flight_sustained_opponent_boundary.py"),
    )
    candidate = Path(a.candidate)
    candidate_result = read(candidate / "result.json")
    candidate_hash = sha256(candidate / "temporal-boundary.pt")
    source = read(Path(plan["source"]) / "result.json")
    root = Path(plan["data"])
    dataset = read(root / "dataset.json")
    entries = [e for e in dataset["episodes"] if e["role"] == "train"]
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or candidate_hash != candidate_result["checkpoint_sha256"]
        or candidate_result["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or candidate_result["dataset_sha256"] != sha256(root / "dataset.json")
        or len(entries) != 6
        or not torch.cuda.is_available()
    ):
        raise ValueError("Exact fixed graph, six TRAIN histories and GPU required")
    if a.mode == "train":
        if not a.qualification:
            raise ValueError("Exact train-program qualification required")
        proof = read(Path(a.qualification) / "result.json")
        if (
            proof["status"] != "qualified_sustained_motion_supervised_gradient"
            or proof["program_sha256"] != sha256(__file__)
            or proof["boundary_source_sha256"] != sha256(module)
            or proof["parent_boundary_sha256"]
            != sha256("src/flybrain_drive/flight_opponent_temporal_boundary.py")
            or proof["candidate_checkpoint_sha256"] != candidate_hash
            or proof["dataset_sha256"] != sha256(root / "dataset.json")
            or proof["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        ):
            raise ValueError("Qualification must match exact training inputs and program")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Original graph and model contract required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    temporal = SustainedOpponentBoundary(
        model.backbone.scene_command.mean,
        model.backbone.scene_command.scale,
        len(model.backbone.vision_ports),
    ).to("cuda")
    saved = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    if saved["actor_qualified"] is not False:
        raise ValueError("Research-only candidate required")
    missing, unexpected = temporal.load_state_dict(saved["model"], strict=False)
    if missing != ["sustained_projection.weight"] or unexpected:
        raise ValueError("Only the new slow projection may be missing")
    temporal.opponent_projection.requires_grad_(False)
    direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    if torch.isin(direct, model.backbone.vision_ports).any():
        raise ValueError("Separate visual and descending ports required")
    before = {key: tensor_digest(value) for key, value in model.state_dict().items()}
    chosen = entries[:1] if a.mode == "qualify" else entries
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "mode": a.mode,
            "boundary_source_sha256": sha256(module),
            "parent_boundary_sha256": sha256(
                "src/flybrain_drive/flight_opponent_temporal_boundary.py"
            ),
            "candidate_checkpoint_sha256": candidate_hash,
            "source_checkpoint_sha256": tr["checkpoint_sha256"],
            "dataset_sha256": sha256(root / "dataset.json"),
            "original_train_ids": [e["file"] for e in chosen],
            "tbptt_observations": 64,
            "optimizer": "AdamW lr=.001 weight_decay=.002",
            "update_schedule": "one update per episode; fixed weights during replay",
            "teacher": "original masked signed steering and forward targets; loss only",
            "direct_motion": "original full four-view motion",
            "time_constants_seconds": [0.05, 0.2],
            "opponent_direction_pairs": "T4b-a,c-d and T5b-a,c-d per view",
            "actor_deployed": False,
            "development_opened": False,
            "seed": None,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    optimizer = torch.optim.AdamW(
        temporal.sustained_projection.parameters(), lr=0.001, weight_decay=0.002
    )
    started = time.perf_counter()
    reports, update_count, gradients = [], 0, []
    names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
    for entry in chosen:
        path = root / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Original TRAIN history changed")
        with load_array(path, allow_pickle=False) as x:
            times = x["observation_time"].copy()
            data = {key: x[key].copy() for key in (*names, "target", "target_valid")}
        complete_history(times, times)
        if (
            len(times) != entry["observations"]
            or data["visual_valid"].any()
            or data["target_valid"].dtype != bool
            or not np.isfinite(data["target"]).all()
            or np.any(np.abs(data["target"]) > 1)
        ):
            raise ValueError("Exact original source, masks and bounded targets required")
        t = {key: torch.tensor(data[key], device="cuda") for key in data}
        with torch.no_grad():
            base = model._drive_tensor(*(t[key] for key in names))
        stop = min(64, len(times)) if a.mode == "qualify" else len(times)
        global_valid = int(t["target_valid"][:stop].sum())
        if global_valid == 0:
            raise ValueError("Nonempty original supervised labels required")
        optimizer.zero_grad(set_to_none=True)
        original_weight = temporal.sustained_projection.weight.detach().clone()
        state, held, traces = model.initial_state(1), None, None
        losses, windows = [], 0
        for start in range(0, stop, 64):
            end = min(start + 64, stop)
            pred, idx = [], []
            for i in range(start, end):
                elapsed = None if i == 0 else float(times[i] - times[i - 1])
                if i:
                    for dt in substeps(elapsed):
                        state = model._advance_elapsed_tensor(held, state, dt)
                if t["target_valid"][i].any():
                    pred.append(model._read_state_preactivation_tensor(state)[0][0])
                    idx.append(i)
                delta, traces = temporal.step(
                    t["scene_features"][i : i + 1],
                    torch.ones((1, 4), dtype=torch.bool, device="cuda"),
                    traces,
                    elapsed,
                )
                held = temporal.modulate(base[i : i + 1], model.backbone.vision_ports, delta)
            if pred:
                predicted = torch.stack(pred)
                target = (t["target"][idx] + 1) / 2
                mask = t["target_valid"][idx]
                loss = (
                    F.binary_cross_entropy_with_logits(2 * predicted, target, reduction="none")
                    * mask
                ).sum() / global_valid
                if not torch.isfinite(loss):
                    raise ValueError("Finite original-label loss required")
                if loss.requires_grad:
                    loss.backward()
                    losses.append(float(loss.detach()))
                    windows += 1
            state, held = state.detach(), held.detach()
            traces = tuple(x.detach() for x in traces)
            if not torch.equal(temporal.sustained_projection.weight.detach(), original_weight):
                raise ValueError("No parameters may change during a complete episode")
        gradient = temporal.sustained_projection.weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or not torch.any(gradient != 0):
            raise ValueError("Nonzero finite accumulated temporal gradient required")
        norm = float(gradient.norm())
        gradients.append(norm)
        if a.mode == "train":
            optimizer.step()
            update_count += 1
        if sha256(path) != entry["sha256"]:
            raise ValueError("Original TRAIN history changed during processing")
        reports.append(
            {
                "file": entry["file"],
                "rows": stop,
                "label_cells": global_valid,
                "gradient_windows": windows,
                "accumulated_loss": sum(losses),
                "gradient_norm": norm,
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    if any(tensor_digest(value) != before[key] for key, value in model.state_dict().items()):
        raise ValueError("Original graph, body, direct pathways or readout changed")
    result = {
        "status": (
            "qualified_sustained_motion_supervised_gradient"
            if a.mode == "qualify"
            else "completed_sustained_motion_boundary_onepass"
        ),
        "program_sha256": sha256(__file__),
        "boundary_source_sha256": sha256(module),
        "parent_boundary_sha256": sha256("src/flybrain_drive/flight_opponent_temporal_boundary.py"),
        "candidate_checkpoint_sha256": candidate_hash,
        "dataset_sha256": sha256(root / "dataset.json"),
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "reports": reports,
        "optimizer_updates": update_count,
        "original_model_unchanged": True,
        "direct_motion_held_mean": False,
        "development_opened": False,
        "physiology_or_driving_verified": False,
        "actor_deployed": False,
        "goal_achieved": False,
        "wall_seconds": time.perf_counter() - started,
    }
    if a.mode == "train":
        if update_count != len(entries):
            raise ValueError("One update per complete TRAIN history required")
        torch.save(
            {
                "schema": "flybrain.sustained_opponent_motion_boundary.v1",
                "model": {k: v.detach().cpu() for k, v in temporal.state_dict().items()},
                "time_constants_seconds": [0.05, 0.2],
                "candidate_checkpoint_sha256": candidate_hash,
                "source_checkpoint_sha256": tr["checkpoint_sha256"],
                "actor_qualified": False,
            },
            run / "temporal-boundary.pt",
        )
        result["checkpoint_sha256"] = sha256(run / "temporal-boundary.pt")
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
