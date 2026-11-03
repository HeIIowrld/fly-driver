"""Fit temporal sensory boundary to exterior TRAIN future-path side, offline only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history, speed_goals
from flybrain_drive.flight_temporal_motion_boundary import TemporalMotionBoundary
from flybrain_drive.graph import sha256
from scripts.qualify_flight_pedal_graph_v2 import tensor_digest
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "initial", "sensors", "labels", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=("qualify", "train"), required=True)
    p.add_argument("--qualification")
    a = p.parse_args()
    training, initial, sensors, labels = map(Path, (a.training, a.initial, a.sensors, a.labels))
    tr, tm = read(training / "result.json"), read(training / "manifest.json")
    plan = tm["config"]["plan"]
    source = read(Path(plan["source"]) / "result.json")
    sr, lr = read(sensors / "result.json"), read(labels / "result.json")
    initial_result = read(initial / "result.json")
    module = "src/flybrain_drive/flight_temporal_motion_boundary.py"
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or sr["status"] != "completed_spatial_flight_sensor_replay"
        or lr["status"] != "completed_exterior_flight_preview_dataset"
        or initial_result["status"] != "completed_temporal_motion_boundary_onepass"
        or initial_result["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or initial_result["dataset_sha256"] != sha256(Path(plan["data"]) / "dataset.json")
        or initial_result["boundary_source_sha256"] != sha256(module)
        or sha256(initial / "temporal-boundary.pt") != initial_result["checkpoint_sha256"]
    ):
        raise ValueError("Exact original fixed graph and exterior datasets required")
    if a.mode == "train":
        if not a.qualification:
            raise ValueError("Exact-source gradient qualification required")
        proof = read(Path(a.qualification) / "result.json")
        if (
            proof["status"] != "qualified_exterior_temporal_preview_gradient"
            or proof["program_sha256"] != sha256(__file__)
            or proof["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or proof["labels_dataset_sha256"] != sha256(labels / "dataset.json")
            or proof["sensors_result_sha256"] != sha256(sensors / "result.json")
            or proof["initial_temporal_sha256"] != initial_result["checkpoint_sha256"]
            or proof["boundary_source_sha256"] != sha256(module)
        ):
            raise ValueError("Pinned successful gradient qualification required")
    ls = [e for e in read(labels / "dataset.json")["episodes"] if e["role"] == "train"]
    ss = {e["id"]: e for e in sr["episodes"] if e["role"] == "train"}
    if len(ls) != 3 or set(ss) != {e["id"] for e in ls}:
        raise ValueError("Three original TRAIN roles required")
    # Training labels are used only in the loss, never in sensor input or drive.
    label_cache, counts = {}, np.zeros(2, dtype=np.int64)
    for e in ls:
        if sha256(e["targets_file"]) != e["targets_sha256"]:
            raise ValueError("Original teacher preview labels changed")
        with load_array(e["targets_file"], allow_pickle=False) as x:
            y, valid, times = (
                x["preview"].copy(),
                x["preview_valid"].copy(),
                x["observation_time"].copy(),
            )
        if (
            y.shape != (len(times), 3, 2)
            or valid.shape != (len(times), 3)
            or valid.dtype != bool
            or not np.isfinite(y).all()
        ):
            raise ValueError("Complete finite teacher labels and masks required")
        curvature = y[:, 2, 1]
        active = valid[:, 2] & (np.abs(curvature) > 0.001)
        counts += [(active & (curvature > 0)).sum(), (active & (curvature < 0)).sum()]
        label_cache[e["id"]] = (curvature, active, times)
    if (counts == 0).any():
        raise ValueError("Both teacher-path directions required in TRAIN")
    weights = counts.sum() / (2 * counts.astype(float))
    if not torch.cuda.is_available():
        raise ValueError("Original GPU required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["spec"] != model.spec or cp["schema"] != model.schema:
        raise ValueError("Exact original graph specification required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    temporal = TemporalMotionBoundary(
        model.backbone.scene_command.mean,
        model.backbone.scene_command.scale,
        len(model.backbone.vision_ports),
    ).to("cuda")
    initial_payload = torch.load(
        initial / "temporal-boundary.pt", map_location="cpu", weights_only=True
    )
    if (
        initial_payload["schema"] != "flybrain.temporal_motion_boundary.v1"
        or initial_payload["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or initial_payload["actor_qualified"] is not False
        or initial_payload["time_constants_seconds"] != [0.05, 0.2]
    ):
        raise ValueError("Exact initial unqualified temporal model required")
    temporal.load_state_dict(initial_payload["model"], strict=True)
    if not torch.equal(temporal.mean, model.backbone.scene_command.mean) or not torch.equal(
        temporal.scale, model.backbone.scene_command.scale
    ):
        raise ValueError("Original TRAIN normalization required")
    trainable = list(temporal.parameters())
    if len(trainable) != 1 or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Only temporal sensory projection may train")
    before = {k: tensor_digest(v) for k, v in model.state_dict().items()}
    spec = read(plan["source_spec"])
    if sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]:
        raise ValueError("Original model scene registration changed")
    actual = ls[:1] if a.mode == "qualify" else ls
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "source_checkpoint_sha256": tr["checkpoint_sha256"],
            "sensors_result_sha256": sha256(sensors / "result.json"),
            "initial_temporal_sha256": initial_result["checkpoint_sha256"],
            "boundary_source_sha256": sha256(module),
            "labels_dataset_sha256": sha256(labels / "dataset.json"),
            "mode": a.mode,
            "training_ids": [e["id"] for e in actual],
            "train_sign_counts": counts.tolist(),
            "class_weights": weights.tolist(),
            "curvature_threshold_1_m": 0.001,
            "teacher_scope": "20m future path side used only as training target; not actor input",
            "goal_kmh": 30,
            "recorded_teacher_goal_kmh": 60,
            "counterfactual_goal": True,
            "direct_motion_ports": "TRAIN mean",
            "tbptt_observations": 64,
            "lr": 0.001,
            "update_schedule": "one fixed-weight whole-episode replay per TRAIN history",
            "temporal_initialization": "prior original six-TRAIN temporal checkpoint",
            "epochs": 1,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.002)
    direct = torch.cat((model.backbone.steering_ports, model.backbone.goal_ports))
    updates, gradients, summaries = 0, [], []
    for le in actual:
        e = ss[le["id"]]
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
        ):
            raise ValueError("Exact original aligned sensory evidence required")
        receipt = read(record / "control/api-receipt.json")
        ai_rows = [
            row
            for row in receipt["telemetry"]
            if row["phase"] == "driving" and row["ai_controlled"] is True
        ]
        if not ai_rows or any(row.get("teacher_speed_hint_kmh") != 60 for row in ai_rows):
            raise ValueError("Recorded 60 km/h teacher task required")
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
            load_array(body / "virtual-body.npz", allow_pickle=False) as b,
        ):
            times, body_times = (
                x["decision_monotonic_seconds"].copy(),
                b["decision_monotonic_seconds"].copy(),
            )
            np.testing.assert_array_equal(x["phases"], b["phases"])
            scene = np.concatenate(
                (x["retina_intensity"][:, :, None], x["motion_activity"]), axis=2
            )
            body_values, body_valid = b["values"].copy(), b["valid"].copy()
        complete_history(times, body_times)
        curvature, active, label_times = label_cache[e["id"]]
        np.testing.assert_array_equal(times, label_times)
        if (
            scene.shape != (len(times), 4, 9, 721)
            or body_values.shape != (len(times), 21)
            or body_valid.shape != body_values.shape
            or body_valid.dtype != bool
            or not np.isfinite(scene).all()
            or not np.isfinite(body_values).all()
            or not body_valid[:, [0, 1, 4, 16, 17, 18, 19, 20]].all()
        ):
            raise ValueError("Complete finite sensor histories and original masks required")
        n = len(times)
        with torch.no_grad():
            scene_t = model.backbone.scene_pool(
                torch.tensor(scene, dtype=torch.float32, device="cuda")
            )
        del scene
        inp = (
            torch.zeros((n, 6), device="cuda"),
            torch.zeros((n, 6), dtype=torch.bool, device="cuda"),
            torch.tensor(body_values, dtype=torch.float32, device="cuda"),
            torch.tensor(body_valid, device="cuda"),
            torch.tensor(speed_goals(30, n), device="cuda"),
            scene_t,
        )
        schedule = [[]] + [substeps(float(t)) for t in np.diff(times)]
        stop = min(64, n) if a.mode == "qualify" else n
        total_active = int(active[:stop].sum())
        if not total_active:
            raise ValueError("Original nonempty active teacher labels required")
        with torch.no_grad():
            base = model._drive_tensor(*inp)
            mean_scene = scene_t.clone()
            mask = torch.zeros((4, 9, 16), dtype=torch.bool, device="cuda")
            mask[:, 1:] = True
            mask = mask.flatten()
            mean_scene[:, mask] = model.backbone.scene_command.mean[mask]
            mean_drive = model._drive_tensor(*inp[:5], mean_scene)
            base[:, direct] = mean_drive[:, direct]
        optimizer.zero_grad(set_to_none=True)
        before_weight = temporal.projection.weight.detach().clone()
        state, held, traces, episode_updates = model.initial_state(1), None, None, 0
        for start in range(0, stop, 64):
            end = min(start + 64, stop)
            selected, outputs = [], []
            for i in range(start, end):
                elapsed = None if i == 0 else float(times[i] - times[i - 1])
                if i:
                    for dt in schedule[i]:
                        state = model._advance_elapsed_tensor(held, state, dt)
                if active[i]:
                    selected.append(i)
                    outputs.append(model._read_state_preactivation_tensor(state)[0][0, 0])
                delta, traces = temporal.step(
                    scene_t[i : i + 1],
                    torch.ones((1, 4), dtype=torch.bool, device="cuda"),
                    traces,
                    elapsed,
                )
                held = temporal.modulate(base[i : i + 1], model.backbone.vision_ports, delta)
            if outputs:
                logits = 2 * torch.stack(outputs)
                signs = curvature[selected] < 0
                target = torch.tensor(signs, device="cuda", dtype=torch.float32)
                weight = torch.tensor(
                    np.where(signs, weights[1], weights[0]), device="cuda", dtype=torch.float32
                )
                loss = (
                    F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight
                ).sum() / total_active
                if not torch.isfinite(loss):
                    raise ValueError("Finite original teacher-side loss required")
                if loss.requires_grad:
                    loss.backward()
            state, held = state.detach(), held.detach()
            traces = tuple(x.detach() for x in traces)
            if not torch.equal(temporal.projection.weight.detach(), before_weight):
                raise ValueError("Temporal weights must stay fixed within each history")
        grad = temporal.projection.weight.grad
        if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
            raise ValueError("Nonzero finite supervised temporal gradient required")
        gradients.append(float(grad.norm()))
        if a.mode == "train":
            optimizer.step()
            updates += 1
            episode_updates = 1
        summaries.append(
            {
                "id": e["id"],
                "observations": n,
                "replayed": stop,
                "active_targets": int(active[:stop].sum()),
                "updates": episode_updates,
                "teacher_receipt_sha256": sha256(record / "control/api-receipt.json"),
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
    after = {k: tensor_digest(v) for k, v in model.state_dict().items()}
    if before != after or not gradients or not any(g > 0 for g in gradients):
        raise ValueError("Original fixed model and nonzero temporal gradient required")
    result = {
        "status": (
            "qualified_exterior_temporal_preview_gradient"
            if a.mode == "qualify"
            else "completed_exterior_temporal_preview_train"
        ),
        "program_sha256": sha256(__file__),
        "boundary_source_sha256": sha256(module),
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "labels_dataset_sha256": sha256(labels / "dataset.json"),
        "sensors_result_sha256": sha256(sensors / "result.json"),
        "initial_temporal_sha256": initial_result["checkpoint_sha256"],
        "training": summaries,
        "optimizer_updates": updates,
        "gradient_norm_range": [min(gradients), max(gradients)],
        "teacher_used_as_input": False,
        "positive_brake_labels_used": 0,
        "original_model_unchanged": True,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    if a.mode == "train":
        if updates != 3:
            raise ValueError("Exactly one optimizer update per original TRAIN episode required")
        torch.save(
            {
                "schema": "flybrain.temporal_motion_boundary.v1",
                "model": {k: v.detach().cpu() for k, v in temporal.state_dict().items()},
                "time_constants_seconds": [0.05, 0.2],
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
