"""Compare temporal credit horizons at a fixed visual flight boundary, without optimization."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    training, candidate = Path(a.training), Path(a.candidate)
    receipt, plan = (
        read(training / "result.json"),
        read(training / "manifest.json")["config"]["plan"],
    )
    source, cand = read(Path(plan["source"]) / "result.json"), read(candidate / "result.json")
    if (
        receipt["status"] != "completed_fixed_joint_command_training"
        or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or cand["status"] != "completed_one_pass_hsvs_boundary_train"
        or cand["source_checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != cand["checkpoint_sha256"]
    ):
        raise ValueError("Original fixed checkpoint and exact learned boundary required")
    if not torch.cuda.is_available():
        raise ValueError("Original GPU execution required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    checkpoint = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["spec"] != model.spec or checkpoint["schema"] != model.schema:
        raise ValueError("Original model specification required")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    visual = model.backbone.vision
    boundary = torch.load(candidate / "visual-boundary.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        visual.weight.copy_(boundary["weight"].to("cuda"))
        visual.bias.copy_(boundary["bias"].to("cuda"))
    visual.requires_grad_(True)
    direct = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    if torch.isin(direct, model.backbone.vision_ports).any():
        raise ValueError("Direct and visual port sets must be disjoint")
    root = Path(plan["data"])
    episodes = [e for e in read(root / "dataset.json")["episodes"] if e["role"] == "train"]
    if len(episodes) != 6:
        raise ValueError("Six original training histories required")
    entry = episodes[0]
    if sha256(root / entry["file"]) != entry["sha256"]:
        raise ValueError("Original train history changed")
    with load_array(root / entry["file"], allow_pickle=False) as x:
        arrays = {
            k: x[k][:128].copy()
            for k in (
                "visual",
                "visual_valid",
                "body",
                "body_valid",
                "goals",
                "scene_features",
                "target",
                "target_valid",
                "observation_time",
            )
        }
    if (
        len(arrays["observation_time"]) != 128
        or arrays["visual_valid"].any()
        or not np.isfinite(arrays["target"]).all()
        or np.any(np.abs(arrays["target"]) > 1)
    ):
        raise ValueError("Original 128-row source and masks required")
    names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
    data = {k: torch.tensor(v, device="cuda") for k, v in arrays.items() if k != "observation_time"}
    schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays["observation_time"])]
    total_valid = int(data["target_valid"].sum().item())
    if total_valid == 0:
        raise ValueError("Original valid training labels required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "source_checkpoint_sha256": receipt["checkpoint_sha256"],
            "visual_boundary_sha256": cand["checkpoint_sha256"],
            "dataset_sha256": sha256(root / "dataset.json"),
            "episode_sha256": entry["sha256"],
            "train_episode": entry["file"],
            "original_rows": 128,
            "horizons": [8, 32, 64],
            "loss": "original BCE weighted by global valid head count; parameters never updated",
            "direct_motion": "fixed to TRAIN mean",
            "actor_deployed": False,
            "optimizer_updates": 0,
            "development_opened": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    reports, arrays_out = {}, {}
    for window in (8, 32, 64):
        visual.zero_grad(set_to_none=True)
        state, held, predictions = model.initial_state(1), None, []
        total_loss = 0.0
        for start in range(0, 128, window):
            end = min(start + window, 128)
            inputs = [data[k][start:end] for k in names]
            drive = model._drive_tensor(*inputs)
            mean_scene = inputs[-1].clone()
            select = torch.zeros(576, dtype=torch.bool, device="cuda").reshape(4, 9, 16)
            select[:, 1:] = True
            select = select.flatten()
            mean_scene[:, select] = model.backbone.scene_command.mean[select]
            with torch.no_grad():
                mean_drive = model._drive_tensor(*inputs[:5], mean_scene)
            drive = drive.clone()
            drive[:, direct] = mean_drive[:, direct]
            chunk = []
            for index in range(start, end):
                if index:
                    for dt in schedule[index]:
                        state = model._advance_elapsed_tensor(held, state, dt)
                chunk.append(model._read_state_preactivation_tensor(state)[0][0])
                held = drive[index - start : index - start + 1]
            pred = torch.stack(chunk)
            predictions.append(pred.detach().cpu().numpy())
            target = (data["target"][start:end] + 1) / 2
            valid = data["target_valid"][start:end]
            loss = (
                F.binary_cross_entropy_with_logits(2 * pred, target, reduction="none") * valid
            ).sum() / total_valid
            if not torch.isfinite(loss):
                raise ValueError("Finite supervised loss required")
            total_loss += float(loss.detach())
            if loss.requires_grad:
                loss.backward()
            state, held = state.detach(), held.detach()
        if visual.weight.grad is None or visual.bias.grad is None:
            raise ValueError("Visual boundary must receive finite gradients")
        grad = torch.cat([visual.weight.grad.flatten(), visual.bias.grad.flatten()]).cpu().numpy()
        if not np.isfinite(grad).all():
            raise ValueError("Finite gradients required")
        arrays_out["gradient_" + str(window)] = grad
        arrays_out["prediction_" + str(window)] = np.concatenate(predictions)
        reports[str(window)] = {"loss": total_loss, "gradient_norm": float(np.linalg.norm(grad))}
    for window in (32, 64):
        error = float(
            np.max(np.abs(arrays_out["prediction_8"] - arrays_out["prediction_" + str(window)]))
        )
        if error > 1e-6:
            raise ValueError("Gradient detach changed forward numerical state")
        g8, g = arrays_out["gradient_8"], arrays_out["gradient_" + str(window)]
        norm = np.linalg.norm(g8) * np.linalg.norm(g)
        reports[str(window)]["forward_max_error_from_8"] = error
        reports[str(window)]["gradient_cosine_with_8"] = float(g8 @ g / norm) if norm else None
    np.savez_compressed(run / "traces.npz", **arrays_out)
    write_json(
        run / "result.json",
        {
            "status": "completed_visual_credit_horizon_audit",
            "reports": reports,
            "trace_sha256": sha256(run / "traces.npz"),
            "actor_deployed": False,
            "optimizer_updates": 0,
            "physiology_validated": False,
            "goal_achieved": False,
        },
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
