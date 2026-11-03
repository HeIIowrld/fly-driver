"""TRAIN-only gain selection for learned visual motion; DEV is evaluated once offline."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history, speed_goals
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "sensors", "labels", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    training, candidate, sensors, labels = map(Path, (a.training, a.candidate, a.sensors, a.labels))
    tr, tm = read(training / "result.json"), read(training / "manifest.json")
    plan = tm["config"]["plan"]
    sr, cr = read(sensors / "result.json"), read(candidate / "result.json")
    source = read(Path(plan["source"]) / "result.json")
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or cr["status"] != "completed_exterior_motion_preview_train"
        or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or cr["sensors_result_sha256"] != sha256(sensors / "result.json")
        or cr["labels_dataset_sha256"] != sha256(labels / "dataset.json")
        or sha256(candidate / "visual-boundary.pt") != cr["checkpoint_sha256"]
    ):
        raise ValueError("Exact original motion candidate and sensory sources required")
    ls = read(labels / "dataset.json")["episodes"]
    sensor_entries = {e["id"]: e for e in sr["episodes"]}
    train = [e for e in ls if e["role"] == "train"]
    dev = [e for e in ls if e["role"] == "development"]
    if (
        len(train) != 3
        or len(dev) != 1
        or {e["id"] for e in ls} != set(sensor_entries)
        or any(sensor_entries[e["id"]]["role"] != e["role"] for e in ls)
    ):
        raise ValueError("Exact source TRAIN and DEVELOPMENT roles required")
    if not torch.cuda.is_available():
        raise ValueError("Original GPU required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    payload = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if payload["spec"] != model.spec or payload["schema"] != model.schema:
        raise ValueError("Exact original fixed graph specification required")
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    original_w = model.backbone.vision.weight.detach().clone()
    original_b = model.backbone.vision.bias.detach().clone()
    boundary = torch.load(candidate / "visual-boundary.pt", map_location="cpu", weights_only=True)
    dw = boundary["weight"].to("cuda") - original_w
    db = boundary["bias"].to("cuda") - original_b
    mask = torch.zeros(4, 9, 16, dtype=torch.bool, device="cuda")
    mask[:, 1:] = True
    mask = mask.flatten()
    if (
        torch.count_nonzero(dw[:, :12])
        or torch.count_nonzero(dw[:, 12:][:, ~mask])
        or (db + dw[:, 12:] @ model.backbone.scene_command.mean).abs().max() > 1e-6
    ):
        raise ValueError("Candidate must preserve non-motion and mean-motion input")
    spec = read(plan["source_spec"])
    if sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]:
        raise ValueError("Original model scene registration changed")
    gains = [0.0, 1.0, 8.0, 32.0, 128.0]
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "source_checkpoint_sha256": tr["checkpoint_sha256"],
            "candidate_sha256": cr["checkpoint_sha256"],
            "sensors_result_sha256": sha256(sensors / "result.json"),
            "labels_dataset_sha256": sha256(labels / "dataset.json"),
            "predeclared_gains": gains,
            "selection": "lowest class-balanced cross-entropy on three TRAIN histories only",
            "goal_kmh": 30,
            "source_teacher_goal_kmh": 60,
            "goal_is_counterfactual": True,
            "direct_motion_ports": "TRAIN mean in all conditions",
            "target": "future 20m path side only after each inference",
            "threshold_1_m": 0.001,
            "known_development_not_blind": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    direct = torch.cat((model.backbone.steering_ports, model.backbone.goal_ports))
    saved, proofs = {}, []

    def replay(le, values):
        e = sensor_entries[le["id"]]
        vision, body, record = map(Path, (e["output"], e["body_replay"], e["recording"]))
        vr, br, vm = (
            read(vision / "result.json"),
            read(body / "result.json"),
            read(vision / "manifest.json")["config"],
        )
        if (
            sha256(vision / "result.json") != e["result_sha256"]
            or sha256(body / "result.json") != e["body_result_sha256"]
            or sha256(vision / "responses.npz") != e["responses_sha256"]
            or vr["responses_sha256"] != e["responses_sha256"]
            or sha256(body / "virtual-body.npz") != br["virtual_body_sha256"]
            or br["contract_sha256"] != sha256("configs/ac-virtual-navigation-body-v2.json")
            or vm["events_sha256"] != sha256(record / "events.jsonl")
            or sha256(le["targets_file"]) != le["targets_sha256"]
        ):
            raise ValueError("Original visual, body, labels and recording hashes required")
        with (
            load_array(spec["scene_registration_file"], allow_pickle=False) as old,
            load_array(vision / "retina-registration.npz", allow_pickle=False) as reg,
        ):
            for key in (
                "u",
                "v",
                "image_uv",
                "body_directions",
                "view_order",
                "motion_types",
                "motion_indices",
            ):
                np.testing.assert_array_equal(old[key], reg[key])
        with (
            load_array(vision / "responses.npz", allow_pickle=False) as x,
            load_array(body / "virtual-body.npz", allow_pickle=False) as y,
        ):
            times, bt = (
                x["decision_monotonic_seconds"].copy(),
                y["decision_monotonic_seconds"].copy(),
            )
            np.testing.assert_array_equal(x["phases"], y["phases"])
            scene = np.concatenate(
                (x["retina_intensity"][:, :, None], x["motion_activity"]), axis=2
            )
            bv, bm = y["values"].copy(), y["valid"].copy()
        complete_history(times, bt)
        n = len(times)
        if (
            scene.shape != (n, 4, 9, 721)
            or bv.shape != (n, 21)
            or bm.shape != bv.shape
            or bm.dtype != bool
            or not np.isfinite(scene).all()
            or not np.isfinite(bv).all()
            or not bm[:, [0, 1, 4, 16]].all()
        ):
            raise ValueError("Valid original four-view and masked body histories required")
        with torch.no_grad():
            features = model.backbone.scene_pool(
                torch.tensor(scene, dtype=torch.float32, device="cuda")
            )
            body_t = torch.tensor(bv, dtype=torch.float32, device="cuda")
            valid_t = torch.tensor(bm, device="cuda")
            goals = torch.tensor(speed_goals(30, n), device="cuda")
            visual = torch.zeros((n, 6), device="cuda")
            vmask = torch.zeros((n, 6), dtype=torch.bool, device="cuda")
            mean_scene = features.clone()
            mean_scene[:, mask] = model.backbone.scene_command.mean[mask]
            inputs = (visual, vmask, body_t, valid_t, goals)
            drives = []
            for gain in values:
                model.backbone.vision.weight.copy_(original_w + gain * dw)
                model.backbone.vision.bias.copy_(original_b + gain * db)
                drive = model._drive_tensor(*inputs, features)
                mean_drive = model._drive_tensor(*inputs, mean_scene)
                drive[:, direct] = mean_drive[:, direct]
                drives.append(drive)
            schedule = [[]] + [substeps(float(dt)) for dt in np.diff(times)]
            state, held, outputs = model.initial_state(len(values)), None, []
            for i, intervals in enumerate(schedule):
                if i:
                    for dt in intervals:
                        state = model._advance_elapsed_tensor(held, state, dt)
                outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
                held = torch.cat([drive[i : i + 1] for drive in drives])
        output = np.stack(outputs)
        # Teacher path never enters the actor; labels are opened after fixed inference.
        with load_array(le["targets_file"], allow_pickle=False) as y:
            np.testing.assert_array_equal(times, y["observation_time"])
            curv, valid = y["preview"][:, 2, 1].copy(), y["preview_valid"][:, 2].copy()
        active = valid & (np.abs(curv) > 0.001)
        if (
            valid.dtype != bool
            or not np.isfinite(curv).all()
            or not np.isfinite(output).all()
            or np.any(np.abs(output) >= 1)
        ):
            raise ValueError("Original target and bounded model outputs required")
        saved[le["id"] + "_output"] = output
        saved[le["id"] + "_curvature"] = curv
        saved[le["id"] + "_active"] = active
        proofs.append(
            {
                "id": le["id"],
                "role": le["role"],
                "vision_sha256": vr["responses_sha256"],
                "body_sha256": br["virtual_body_sha256"],
                "label_sha256": le["targets_sha256"],
            }
        )
        print(
            json.dumps({"id": le["id"], "conditions": len(values), "active": int(active.sum())}),
            flush=True,
        )
        return output, curv, active

    per_gain = {g: {1: [], -1: []} for g in gains}
    for entry in train:
        output, curv, active = replay(entry, gains)
        for j, gain in enumerate(gains):
            for side in (1, -1):
                selected = active & (np.sign(curv) == side)
                probability = (1 - np.sign(curv[selected]) * output[selected, j, 0]) / 2
                per_gain[gain][side].extend((-np.log(np.clip(probability, 1e-7, 1))).tolist())
    losses = {g: float((np.mean(v[1]) + np.mean(v[-1])) / 2) for g, v in per_gain.items()}
    if any(not per_gain[g][side] for g in gains for side in (1, -1)):
        raise ValueError("Both TRAIN target directions required")
    chosen = min(gains, key=lambda g: (losses[g], g))
    # The candidate choice is fixed here before the single already-known DEV is opened.
    write_json(run / "selection.json", {"selected_gain": chosen, "training_loss": losses})
    dv, curvature, active = replay(dev[0], [0.0, chosen])
    results = {}
    for j, name in enumerate(("original", "selected")):
        steering = dv[:, j, 0]
        results[name] = {
            "positive_steer": int((steering > 0).sum()),
            "negative_steer": int((steering < 0).sum()),
            "matching_positive_target": int(((steering < 0) & (curvature > 0) & active).sum()),
            "matching_negative_target": int(((steering > 0) & (curvature < 0) & active).sum()),
            "mean_abs_steering": float(np.abs(steering).mean()),
        }
    np.savez_compressed(run / "replay.npz", **saved)
    write_json(
        run / "result.json",
        {
            "status": "completed_train_selected_motion_gain_diagnostic",
            "selected_gain": chosen,
            "training_class_balanced_loss": losses,
            "development": results,
            "development_positive_targets": int(((curvature > 0) & active).sum()),
            "development_negative_targets": int(((curvature < 0) & active).sum()),
            "proofs": proofs,
            "replay_sha256": sha256(run / "replay.npz"),
            "known_development_not_independent": True,
            "teacher_path_not_optimal_line": True,
            "physical_sync_verified": False,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )
    print(
        json.dumps({"selected_gain": chosen, "training_loss": losses, "development": results}),
        flush=True,
    )


if __name__ == "__main__":
    main()
