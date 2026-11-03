"""Train only VS motion input on teacher future-path side, offline only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_centered_motion_boundary import CenteredMotionBoundary
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history, speed_goals
from flybrain_drive.graph import sha256
from scripts.qualify_flight_pedal_graph_v2 import tensor_digest
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "sensors", "labels", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=("qualify", "train"), required=True)
    p.add_argument("--qualification")
    a = p.parse_args()
    training, sensors, labels = map(Path, (a.training, a.sensors, a.labels))
    tr, tm = read(training / "result.json"), read(training / "manifest.json")
    plan = tm["config"]["plan"]
    source = read(Path(plan["source"]) / "result.json")
    sr, lr = read(sensors / "result.json"), read(labels / "result.json")
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or sr["status"] != "completed_spatial_flight_sensor_replay"
        or lr["status"] != "completed_exterior_flight_preview_dataset"
    ):
        raise ValueError("Exact original fixed graph and exterior datasets required")
    if a.mode == "train":
        if not a.qualification:
            raise ValueError("Exact-source gradient qualification required")
        proof = read(Path(a.qualification) / "result.json")
        if (
            proof["status"] != "qualified_exterior_motion_preview_gradient"
            or proof["program_sha256"] != sha256(__file__)
            or proof["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or proof["labels_dataset_sha256"] != sha256(labels / "dataset.json")
            or proof["sensors_result_sha256"] != sha256(sensors / "result.json")
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
    model.backbone.vision = CenteredMotionBoundary(
        model.backbone.vision, model.backbone.scene_command.mean
    ).to("cuda")
    trainable = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if set(trainable) != {"backbone.vision.delta"}:
        raise ValueError("Only centered visual motion boundary may train")
    spec_for_ports = read(plan['source_spec'])
    if sha256(spec_for_ports['graph_file']) != spec_for_ports['graph_sha256']:
        raise ValueError('Original graph must remain unchanged')
    with load_array(spec_for_ports['graph_file'], allow_pickle=False) as graph:
        graph_types = graph['types'].copy()
    port_ids = model.backbone.vision_ports.detach().cpu().numpy()
    vs_rows = np.char.startswith(graph_types[port_ids], 'VS')
    if int(vs_rows.sum()) != 34 or int((~vs_rows).sum()) != 8:
        raise ValueError('Exactly 34 VS and 8 HS sensory input ports required')
    row_mask = torch.tensor(vs_rows, device='cuda', dtype=torch.float32)[:, None]
    trainable['backbone.vision.delta'].register_hook(lambda gradient: gradient * row_mask)
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
            "lr": 0.005,
            "learned_ports": "VS rows only; HS rows frozen at baseline",
            "epochs": 1,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.005, weight_decay=0.002)
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
            or not body_valid[:, [0, 1, 4, 16]].all()
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
        state, held, episode_updates = model.initial_state(1), None, 0
        for start in range(0, stop, 64):
            end = min(start + 64, stop)
            optimizer.zero_grad(set_to_none=True)
            block = [v[start:end] for v in inp]
            drive = model._drive_tensor(*block)
            mean_scene = block[-1].clone()
            mask = torch.zeros(576, dtype=torch.bool, device="cuda").reshape(4, 9, 16)
            mask[:, 1:] = True
            mask = mask.flatten()
            mean_scene[:, mask] = model.backbone.scene_command.mean[mask]
            with torch.no_grad():
                mean_drive = model._drive_tensor(*block[:5], mean_scene)
            drive = drive.clone()
            drive[:, direct] = mean_drive[:, direct]
            selected, outputs = [], []
            for i in range(start, end):
                if i:
                    for dt in schedule[i]:
                        state = model._advance_elapsed_tensor(held, state, dt)
                if active[i]:
                    selected.append(i)
                    outputs.append(model._read_state_preactivation_tensor(state)[0][0, 0])
                held = drive[i - start : i - start + 1]
            if outputs:
                logits = 2 * torch.stack(outputs)
                # Screen-right positive path means left-positive steering target 0.
                signs = curvature[selected] < 0
                target = torch.tensor(signs, device="cuda", dtype=torch.float32)
                weight = torch.tensor(
                    np.where(signs, weights[1], weights[0]), device="cuda", dtype=torch.float32
                )
                loss = (
                    F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight
                ).mean()
                if not torch.isfinite(loss) or not loss.requires_grad:
                    raise ValueError("Finite connected teacher-side loss required")
                loss.backward()
                grad = trainable["backbone.vision.delta"].grad
                if grad is None or not torch.isfinite(grad).all():
                    raise ValueError("Finite visual boundary gradient required")
                norm = float(grad.norm())
                gradients.append(norm)
                if a.mode == "train" and norm > 0:
                    optimizer.step()
                    updates += 1
                    episode_updates += 1
            state, held = state.detach(), held.detach()
        summaries.append(
            {
                "id": e["id"],
                "observations": n,
                "replayed": stop,
                "active_targets": int(active[:stop].sum()),
                "updates": episode_updates,
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
    if torch.any(trainable['backbone.vision.delta'][~torch.tensor(vs_rows, device='cuda')] != 0):
        raise ValueError('HS rows must remain exactly unchanged')
    after = {k: tensor_digest(v) for k, v in model.state_dict().items()}
    changed = [k for k in before if before[k] != after[k]]
    if (
        set(changed) - {"backbone.vision.delta"}
        or not gradients
        or not any(g > 0 for g in gradients)
    ):
        raise ValueError("Nonzero motion gradient and fixed graph required")
    result = {
        "status": (
            "qualified_exterior_motion_preview_gradient"
            if a.mode == "qualify"
            else "completed_exterior_motion_preview_train"
        ),
        "program_sha256": sha256(__file__),
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "labels_dataset_sha256": sha256(labels / "dataset.json"),
        "sensors_result_sha256": sha256(sensors / "result.json"),
        "training": summaries,
        "optimizer_updates": updates,
        "gradient_norm_range": [min(gradients), max(gradients)],
        "changed_tensors": changed,
        "teacher_used_as_input": False,
        "actor_deployed": False,
        "positive_brake_labels_used": 0,
        "goal_achieved": False,
    }
    if a.mode == "train":
        if updates == 0 or changed != ["backbone.vision.delta"]:
            raise ValueError("Actual motion-only training required")
        weight, bias = model.backbone.vision.materialize()
        torch.save(
            {
                "schema": "flybrain.hsvs_visual_boundary_delta.v1",
                "weight": weight.cpu(),
                "bias": bias.cpu(),
                "source_checkpoint_sha256": tr["checkpoint_sha256"],
                "actor_qualified": False,
            },
            run / "visual-boundary.pt",
        )
        result["checkpoint_sha256"] = sha256(run / "visual-boundary.pt")
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
