"""Fit a supervised engineering VS sensory boundary using saved TRAIN views only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_centered_motion_boundary import CenteredMotionBoundary
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256


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
    tr, plan = read(training / "result.json"), read(training / "manifest.json")["config"]["plan"]
    sr, lr = read(sensors / "result.json"), read(labels / "result.json")
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sr["status"] != "completed_spatial_flight_sensor_replay"
        or lr["status"] != "completed_exterior_flight_preview_dataset"
    ):
        raise ValueError("Original fixed checkpoint and completed saved sensor sources required")
    if a.mode == "train":
        if not a.qualification:
            raise ValueError("Exact-source qualification required")
        proof = read(Path(a.qualification) / "result.json")
        if (
            proof["status"] != "qualified_exterior_vs_sensory_gradient"
            or proof["program_sha256"] != sha256(__file__)
            or proof["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
            or proof["labels_dataset_sha256"] != sha256(labels / "dataset.json")
            or proof["sensors_result_sha256"] != sha256(sensors / "result.json")
        ):
            raise ValueError("Exact qualified original inputs required")
    spec = read(plan["source_spec"])
    if (
        sha256(spec["graph_file"]) != spec["graph_sha256"]
        or sha256(spec["annotations_file"]) != spec["annotations_sha256"]
        or sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]
    ):
        raise ValueError("Original graph, annotations and registration required")
    with load_array(spec["graph_file"], allow_pickle=False) as g:
        ids, types = g["ids"].tolist(), g["types"].copy()
    annotations = {str(row["bodyId"]): row for row in read(spec["annotations_file"])}
    positions = [ids.index(str(i)) for i in spec["vision_input_ids"]]
    sides = np.array([annotations[str(i)]["somaSide"] for i in spec["vision_input_ids"]])
    vs = np.char.startswith(types[positions], "VS")
    left, right = vs & (sides == "L"), vs & (sides == "R")
    if (int(left.sum()), int(right.sum()), int((~vs).sum())) != (16, 18, 8):
        raise ValueError("Original side-labelled VS/HS ports required")
    checkpoint = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if (checkpoint["schema"] != "flybrain.flight.embodied_navigation.v7"
            or checkpoint["spec"]["source_visual_spec_sha256"] != sha256(plan["source_spec"])
            or checkpoint["spec"]["graph_sha256"] != spec["graph_sha256"]):
        raise ValueError("Original embodied and visual graph specification required")
    state = checkpoint["model"]
    weight, bias = state["backbone.vision.weight"], state["backbone.vision.bias"]
    base = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        base.weight.copy_(weight)
        base.bias.copy_(bias)
    boundary = CenteredMotionBoundary(base, state["backbone.scene_command.mean"])
    vs_mask = torch.tensor(vs, dtype=torch.float32)[:, None]
    boundary.delta.register_hook(lambda gradient: gradient * vs_mask)
    with load_array(spec["scene_registration_file"], allow_pickle=False) as x:
        pool = FlightScenePool(x["image_uv"], spec["scene_basis"])
    labels_entries = [e for e in read(labels / "dataset.json")["episodes"] if e["role"] == "train"]
    source_map = {e["id"]: e for e in sr["episodes"] if e["role"] == "train"}
    if len(labels_entries) != 3 or set(source_map) != {e["id"] for e in labels_entries}:
        raise ValueError("Exact three original training histories required")
    torch.set_num_threads(1)
    samples, targets, counts, history = [], [], np.zeros(2, dtype=np.int64), []
    for le in labels_entries:
        e = source_map[le["id"]]
        vision, body, record = map(Path, (e["output"], e["body_replay"], e["recording"]))
        vr, br = read(vision / "result.json"), read(body / "result.json")
        vm = read(vision / "manifest.json")["config"]
        if (
            sha256(vision / "result.json") != e["result_sha256"]
            or sha256(body / "result.json") != e["body_result_sha256"]
            or sha256(vision / "responses.npz") != e["responses_sha256"]
            or vr["responses_sha256"] != e["responses_sha256"]
            or sha256(body / "virtual-body.npz") != br["virtual_body_sha256"]
            or vm["events_sha256"] != sha256(record / "events.jsonl")
            or sha256(le["targets_file"]) != le["targets_sha256"]
        ):
            raise ValueError("Exact original TRAIN evidence required")
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
        with load_array(vision / "responses.npz", allow_pickle=False) as x:
            times = x["decision_monotonic_seconds"].copy()
            scene = np.concatenate(
                (x["retina_intensity"][:, :, None], x["motion_activity"]), axis=2
            )
        with load_array(le["targets_file"], allow_pickle=False) as y:
            np.testing.assert_array_equal(times, y["observation_time"])
            curvature, valid = y["preview"][:, 2, 1].copy(), y["preview_valid"][:, 2].copy()
        if (
            scene.shape != (len(times), 4, 9, 721)
            or valid.dtype != bool
            or not np.isfinite(scene).all()
            or not np.isfinite(curvature).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Complete original ordered visual histories and target masks required")
        active = valid & (np.abs(curvature) > 0.001)
        with torch.no_grad():
            feat = pool(torch.tensor(scene[active], dtype=torch.float32))
        samples.append(torch.cat([torch.zeros((len(feat), 12)), feat], dim=1))
        # VS-left high and VS-right low was positive steering in the artificial probe.
        negative_curve = curvature[active] < 0
        targets.append(torch.tensor(negative_curve, dtype=torch.bool))
        counts += [(active & (curvature > 0)).sum(), (active & (curvature < 0)).sum()]
        history.append(
            {
                "id": e["id"],
                "observations": len(times),
                "active_targets": int(active.sum()),
                "vision_sha256": e["responses_sha256"],
                "label_sha256": le["targets_sha256"],
            }
        )
    if (counts == 0).any():
        raise ValueError("Both TRAIN curvature signs required")
    features, target = torch.cat(samples), torch.cat(targets)
    class_weights = counts.sum() / (2 * counts.astype(float))
    sample_weights = torch.where(
        target,
        torch.tensor(class_weights[1], dtype=torch.float32),
        torch.tensor(class_weights[0], dtype=torch.float32),
    )
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "mode": a.mode,
            "source_checkpoint_sha256": tr["checkpoint_sha256"],
            "sensors_result_sha256": sha256(sensors / "result.json"),
            "labels_dataset_sha256": sha256(labels / "dataset.json"),
            "train_sign_counts": counts.tolist(),
            "train_role_only": True,
            "teacher_target_only": True,
            "target_source": "20m teacher curvature sign",
            "target_vs_current": ("left/right 0.9/0.1 by model steering hypothesis, "
                                  "not fly physiology"),
            "training_steps": 200 if a.mode == "train" else 0,
            "learning_rate": 0.01,
            "source_labels_are_not_optimal_line": True,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    optimizer = torch.optim.AdamW([boundary.delta], lr=0.01, weight_decay=0.002)
    trace = []
    steps = 1 if a.mode == "qualify" else 200
    left_mask, right_mask = torch.tensor(left), torch.tensor(right)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = boundary(features)
        ltarget = torch.where(target, 0.9, 0.1)[:, None].expand(-1, int(left.sum()))
        rtarget = 1 - ltarget[:, :1].expand(-1, int(right.sum()))
        lv = F.binary_cross_entropy_with_logits(
            logits[:, left_mask], ltarget, reduction="none"
        ).mean(1)
        rv = F.binary_cross_entropy_with_logits(
            logits[:, right_mask], rtarget, reduction="none"
        ).mean(1)
        loss = (((lv + rv) / 2) * sample_weights).mean()
        if not torch.isfinite(loss):
            raise ValueError("Finite loss required")
        loss.backward()
        if boundary.delta.grad is None or not torch.isfinite(boundary.delta.grad).all():
            raise ValueError("Finite VS gradient required")
        norm = float(boundary.delta.grad.norm())
        if norm == 0 or torch.any(boundary.delta.grad[~torch.tensor(vs)] != 0):
            raise ValueError("Nonzero VS gradient and frozen HS rows required")
        trace.append(float(loss.detach()))
        if a.mode == "train":
            optimizer.step()
    if torch.any(boundary.delta.detach()[~torch.tensor(vs)] != 0):
        raise ValueError("HS rows must remain frozen")
    result = {
        "status": (
            "qualified_exterior_vs_sensory_gradient"
            if a.mode == "qualify"
            else "completed_exterior_motion_preview_train"
        ),
        "program_sha256": sha256(__file__),
        "source_checkpoint_sha256": tr["checkpoint_sha256"],
        "sensors_result_sha256": sha256(sensors / "result.json"),
        "labels_dataset_sha256": sha256(labels / "dataset.json"),
        "training": history,
        "optimizer_updates": 0 if a.mode == "qualify" else steps,
        "first_last_loss": [trace[0], trace[-1]],
        "last_gradient_norm": norm,
        "teacher_used_as_input": False,
        "actor_deployed": False,
        "physiological_vs_tuning_verified": False,
        "positive_brake_labels_used": 0,
        "goal_achieved": False,
    }
    if a.mode == "train":
        w, b = boundary.materialize()
        torch.save(
            {
                "schema": "flybrain.hsvs_visual_boundary_delta.v1",
                "weight": w,
                "bias": b,
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
