"""Probe bounded HS/VS interventions on a recorded exterior history."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit, modulate_current
from flybrain_drive.flight_learning import complete_history, speed_goals
from flybrain_drive.flight_temporal_motion_boundary import TemporalMotionBoundary
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "sensors", "labels", "prior", "episode", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    training, candidate, sensors, labels, prior = map(
        Path, (a.training, a.candidate, a.sensors, a.labels, a.prior)
    )
    tr, tm, cr, sr, lr, pr = (
        read(training / "result.json"),
        read(training / "manifest.json"),
        read(candidate / "result.json"),
        read(sensors / "result.json"),
        read(labels / "result.json"),
        read(prior / "result.json"),
    )
    plan, prior_config = tm["config"]["plan"], read(prior / "manifest.json")["config"]
    source = read(Path(plan["source"]) / "result.json")
    module = "src/flybrain_drive/flight_temporal_motion_boundary.py"
    root = Path(plan["data"])
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or cr["status"] != "completed_temporal_motion_boundary_onepass"
        or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or cr["boundary_source_sha256"] != sha256(module)
        or cr["dataset_sha256"] != sha256(root / "dataset.json")
        or sha256(candidate / "temporal-boundary.pt") != cr["checkpoint_sha256"]
        or sr["status"] != "completed_spatial_flight_sensor_replay"
        or lr["status"] != "completed_exterior_flight_preview_dataset"
        or pr["status"] != "completed_temporal_exterior_preview_replay"
        or sha256(prior / "predictions.npz") != pr["predictions_sha256"]
        or prior_config["training_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or prior_config["candidate_checkpoint_sha256"] != cr["checkpoint_sha256"]
        or prior_config["episode"] != a.episode
        or not torch.cuda.is_available()
    ):
        raise ValueError("Exact model, temporal candidate, saved sensors and baseline required")
    entries = [e for e in sr["episodes"] if e["id"] == a.episode]
    label_entries = [e for e in read(labels / "dataset.json")["episodes"] if e["id"] == a.episode]
    if (
        len(entries) != 1
        or len(label_entries) != 1
        or entries[0]["role"] != label_entries[0]["role"]
        or entries[0]["role"] != "development"
    ):
        raise ValueError("Exactly one original same-role exterior episode required")
    entry, label = entries[0], label_entries[0]
    vision, body, recording = map(Path, (entry["output"], entry["body_replay"], entry["recording"]))
    vr, br = read(vision / "result.json"), read(body / "result.json")
    vm = read(vision / "manifest.json")["config"]
    if (
        sha256(vision / "result.json") != entry["result_sha256"]
        or sha256(body / "result.json") != entry["body_result_sha256"]
        or sha256(vision / "responses.npz") != entry["responses_sha256"]
        or vr["responses_sha256"] != entry["responses_sha256"]
        or sha256(body / "virtual-body.npz") != br["virtual_body_sha256"]
        or vm["events_sha256"] != sha256(recording / "events.jsonl")
        or sha256(label["targets_file"]) != label["targets_sha256"]
        or br["contract_sha256"] != sha256("configs/ac-virtual-navigation-body-v2.json")
        or prior_config["vision_sha256"] != entry["responses_sha256"]
        or prior_config["body_sha256"] != br["virtual_body_sha256"]
    ):
        raise ValueError("Exact original visual, body, label, event and prior lineage required")
    receipt = recording / "control/api-receipt.json"
    ai = [
        row
        for row in read(receipt)["telemetry"]
        if row["phase"] == "driving" and row["ai_controlled"] is True
    ]
    if not ai or any(row.get("teacher_speed_hint_kmh") != 60 for row in ai):
        raise ValueError("Original 60km/h recorded teacher task required")
    spec = read(plan["source_spec"])
    if sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]:
        raise ValueError("Original visual registration changed")
    with (
        load_array(spec["scene_registration_file"], allow_pickle=False) as old,
        load_array(vision / "retina-registration.npz", allow_pickle=False) as new,
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
            np.testing.assert_array_equal(old[key], new[key])
        origins = new["body_origins_m"].copy()
    with (
        load_array(vision / "responses.npz", allow_pickle=False) as x,
        load_array(body / "virtual-body.npz", allow_pickle=False) as y,
    ):
        times, body_times = (
            x["decision_monotonic_seconds"].copy(),
            y["decision_monotonic_seconds"].copy(),
        )
        np.testing.assert_array_equal(x["phases"], y["phases"])
        scene_raw = np.concatenate(
            (x["retina_intensity"][:, :, None], x["motion_activity"]), axis=2
        )
        body_values, body_valid = y["values"].copy(), y["valid"].copy()
    complete_history(times, body_times)
    if (
        scene_raw.shape != (len(times), 4, 9, 721)
        or body_values.shape != (len(times), 21)
        or body_valid.shape != body_values.shape
        or body_valid.dtype != bool
        or not np.isfinite(scene_raw).all()
        or not np.isfinite(body_values).all()
        or not body_valid[:, [0, 1, 4, 16, 17, 18, 19, 20]].all()
    ):
        raise ValueError("Complete four-view history and required valid body senses required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Exact original v7 model contract required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    tc = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    if (
        tc["schema"] != "flybrain.temporal_motion_boundary.v1"
        or tc["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or tc["actor_qualified"] is not False
        or tc["time_constants_seconds"] != [0.05, 0.2]
    ):
        raise ValueError("Exact fixed research-only temporal candidate required")
    temporal = TemporalMotionBoundary(
        model.backbone.scene_command.mean,
        model.backbone.scene_command.scale,
        len(model.backbone.vision_ports),
    ).to("cuda")
    temporal.load_state_dict(tc["model"], strict=True)
    temporal.eval().requires_grad_(False)
    if not torch.equal(temporal.mean, model.backbone.scene_command.mean) or not torch.equal(
        temporal.scale, model.backbone.scene_command.scale
    ):
        raise ValueError("Original TRAIN-only normalization required")
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        graph_ids, types = graph["ids"].tolist(), graph["types"].copy()
    annotations = {str(row["bodyId"]): row for row in read(spec["annotations_file"])}
    groups = {}
    for family in ("HS", "VS"):
        for side in ("L", "R"):
            positions = [
                graph_ids.index(str(identity))
                for identity in spec["vision_input_ids"]
                if types[graph_ids.index(str(identity))].startswith(family)
                and annotations[str(identity)]["somaSide"] == side
            ]
            groups[family + side] = torch.tensor(positions, device="cuda")
    if [len(groups[k]) for k in ("HSL", "HSR", "VSL", "VSR")] != [4, 4, 16, 18]:
        raise ValueError("Exact original annotated side-labelled visual ports required")
    if (
        sha256(spec["graph_file"]) != spec["graph_sha256"]
        or sha256(spec["annotations_file"]) != spec["annotations_sha256"]
    ):
        raise ValueError("Original graph and annotations required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "training_checkpoint_sha256": tr["checkpoint_sha256"],
            "candidate_checkpoint_sha256": cr["checkpoint_sha256"],
            "episode": entry["id"],
            "original_role": entry["role"],
            "vision_sha256": entry["responses_sha256"],
            "body_sha256": br["virtual_body_sha256"],
            "events_sha256": vm["events_sha256"],
            "teacher_receipt_sha256": sha256(receipt),
            "label_sha256": label["targets_sha256"],
            "prior_predictions_sha256": pr["predictions_sha256"],
            "camera_origins_m": origins.tolist(),
            "source_teacher_goal_kmh": 60,
            "model_goal_kmh": 30,
            "goal_is_counterfactual": True,
            "direct_motion": "paired TRAIN-mean-clamped and original motion paths",
            "target_opened_only_after_inference": True,
            "intervention": "constant +/-0.5 log-odds perturbation, per annotated side",
            "delta_bound_is_temporal_adapter_limit": True,
            "maximal_limit_not_reachable_weight_or_physiology_proof": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    n = len(times)
    with torch.no_grad():
        scene = model.backbone.scene_pool(
            torch.tensor(scene_raw, dtype=torch.float32, device="cuda")
        )
        visual = torch.zeros((n, 6), dtype=torch.float32, device="cuda")
        visual_valid = torch.zeros((n, 6), dtype=torch.bool, device="cuda")
        body_t = torch.tensor(body_values, dtype=torch.float32, device="cuda")
        valid_t = torch.tensor(body_valid, device="cuda")
        goals = torch.tensor(speed_goals(30, n), device="cuda")
        inputs = (visual, visual_valid, body_t, valid_t, goals)
        base = model._drive_tensor(*inputs, scene)
        mean = scene.clone()
        motion = torch.zeros((4, 9, 16), dtype=torch.bool, device="cuda")
        motion[:, 1:] = True
        motion = motion.flatten()
        mean[:, motion] = model.backbone.scene_command.mean[motion]
        mean_drive = model._drive_tensor(*inputs, mean)
        direct = torch.cat((model.backbone.steering_ports, model.backbone.goal_ports))
        original_base = base.clone()
        base[:, direct] = mean_drive[:, direct]
        interventions = (
            "original",
            "vs_left_plus",
            "vs_right_plus",
            "hs_left_plus",
            "hs_right_plus",
            "all_left_plus",
            "all_right_plus",
        )
        case_names = tuple(
            f"{mode}_{name}" for mode in ("clamped", "full") for name in interventions
        )
        drives = [source.clone() for source in (base, original_base) for _ in interventions]
        for i, families, positive_side in (
            (1, ("VS",), "L"),
            (2, ("VS",), "R"),
            (3, ("HS",), "L"),
            (4, ("HS",), "R"),
            (5, ("HS", "VS"), "L"),
            (6, ("HS", "VS"), "R"),
        ):
            for family in families:
                for side in ("L", "R"):
                    ports = groups[family + side]
                    shift = 0.5 if side == positive_side else -0.5
                    for offset, source in ((0, base), (len(interventions), original_base)):
                        drives[i + offset][:, ports] = modulate_current(
                            source[:, ports], torch.full_like(source[:, ports], shift)
                        )
        for index, drive in enumerate(drives):
            source = base if index < len(interventions) else original_base
            if not torch.equal(drive[:, direct], source[:, direct]):
                raise ValueError("Direct descending inputs must remain identical")
        state, held, prediction = model.initial_state(len(case_names)), None, []
        for i in range(n):
            if i:
                for dt in substeps(float(times[i] - times[i - 1])):
                    state = model._advance_elapsed_tensor(held, state, dt)
            prediction.append(model._read_state_tensor(state)[0].cpu().numpy())
            held = torch.cat([drive[i : i + 1] for drive in drives])
        visual_current = base[:, model.backbone.vision_ports].cpu().numpy()
    output = np.stack(prediction)
    with load_array(prior / "predictions.npz", allow_pickle=False) as x:
        np.testing.assert_array_equal(times, x["time"])
        baseline_error = float(np.max(np.abs(output[:, 0] - x["output"][:, 0])))
    if baseline_error > 1e-6:
        raise ValueError("Original direct-clamped baseline must match previous replay")
    # The teacher trajectory is read after all neural predictions have been fixed.
    with load_array(label["targets_file"], allow_pickle=False) as x:
        np.testing.assert_array_equal(times, x["observation_time"])
        preview, valid = x["preview"].copy(), x["preview_valid"].copy()
    if preview.shape != (n, 3, 2) or valid.shape != (n, 3) or valid.dtype != bool:
        raise ValueError("Original signed 20m teacher chord target and mask required")
    curve = preview[:, 2, 1]
    active = valid[:, 2] & (np.abs(curve) > 0.001)
    if not active.any():
        raise ValueError("Nonempty original curvature diagnostic required")
    conditions = {}
    for i, name in enumerate(case_names):
        steering = output[:, i, 0]
        conditions[name] = {
            "positive_steer": int((steering > 0).sum()),
            "negative_steer": int((steering < 0).sum()),
            "heuristic_sign_matches": int(
                (np.sign(steering[active]) == -np.sign(curve[active])).sum()
            ),
            "valid_target_count": int(active.sum()),
            "mean_abs_change_from_mode_original": float(np.mean(np.abs(
                steering - output[:, (i // len(interventions)) * len(interventions), 0]
            ))),
        }
    if len(conditions) != len(case_names) or output.shape[1] != len(case_names):
        raise ValueError("Every intervention must have a recorded result")
    np.savez_compressed(
        run / "predictions.npz",
        output=output,
        time=times,
        curvature=curve,
        target_valid=valid[:, 2],
        active=active,
        baseline_visual_current=visual_current,
    )
    result = {
        "status": "completed_bounded_visual_port_extreme_probe",
        "episode": entry["id"],
        "original_role": entry["role"],
        "conditions": conditions,
        "target_positive_count": int((curve[active] > 0).sum()),
        "target_negative_count": int((curve[active] < 0).sum()),
        "baseline_max_error": baseline_error,
        "full_vs_clamped_original_mean_abs_steering_change": float(np.mean(np.abs(
            output[:, len(interventions), 0] - output[:, 0, 0]
        ))),
        "baseline_visual_current_min_max": [
            float(visual_current.min()),
            float(visual_current.max()),
        ],
        "not_general_reachable_set_bound": True,
        "soma_side_is_not_registered_receptive_field_or_steering_direction": True,
        "predictions_sha256": sha256(run / "predictions.npz"),
        "heuristic_target_not_instantaneous_steering": True,
        "goal_mismatch_counterfactual": True,
        "physical_time_sync_verified": False,
        "known_same_track_not_independent": True,
        "actor_deployed": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
