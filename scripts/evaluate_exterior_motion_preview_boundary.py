"""Replay exterior-preview-trained visual boundary on held-out exterior history."""

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
    source, cr = read(Path(plan["source"]) / "result.json"), read(candidate / "result.json")
    sr, lr = read(sensors / "result.json"), read(labels / "result.json")
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or cr["status"] != "completed_exterior_motion_preview_train"
        or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != cr["checkpoint_sha256"]
        or sr["status"] != "completed_spatial_flight_sensor_replay"
        or lr["status"] != "completed_exterior_flight_preview_dataset"
    ):
        raise ValueError("Exact completed source model and exterior data required")
    if (cr['labels_dataset_sha256'] != sha256(labels/'dataset.json')
            or cr['sensors_result_sha256'] != sha256(sensors/'result.json')
            or any(row['id'].endswith('development') for row in cr['training'])):
        raise ValueError('Candidate must use the same TRAIN-only exterior sources')
    entries = [e for e in sr["episodes"] if e["role"] == "development"]
    targets = [e for e in read(labels / "dataset.json")["episodes"] if e["role"] == "development"]
    if len(entries) != 1 or len(targets) != 1 or entries[0]["id"] != targets[0]["id"]:
        raise ValueError("One same-role exterior development history required")
    entry, label = entries[0], targets[0]
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
    ):
        raise ValueError("Original visual, body, recording and target hashes required")
    if br['contract_sha256'] != sha256('configs/ac-virtual-navigation-body-v2.json'):
        raise ValueError('Original navigation body sensor contract required')
    teacher_receipt = recording/'control/api-receipt.json'
    teacher_rows = [row for row in read(teacher_receipt)['telemetry']
                    if row['phase'] == 'driving' and row['ai_controlled'] is True]
    if not teacher_rows or any(row.get('teacher_speed_hint_kmh') != 60 for row in teacher_rows):
        raise ValueError('Original 60 km/h teacher task must be documented')
    spec = read(plan["source_spec"])
    if sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]:
        raise ValueError("Original model scene registration changed")
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
    if not torch.cuda.is_available():
        raise ValueError("Original GPU execution required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    cp = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    if cp["schema"] != model.schema or cp["spec"] != model.spec:
        raise ValueError("Original graph and checkpoint contracts required")
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    orig_weight = model.backbone.vision.weight.detach().clone()
    orig_bias = model.backbone.vision.bias.detach().clone()
    delta = torch.load(candidate / "visual-boundary.pt", map_location="cpu", weights_only=True)
    with (
        load_array(vision / "responses.npz", allow_pickle=False) as x,
        load_array(body / "virtual-body.npz", allow_pickle=False) as y,
    ):
        times, body_times = (
            x["decision_monotonic_seconds"].copy(),
            y["decision_monotonic_seconds"].copy(),
        )
        np.testing.assert_array_equal(x["phases"], y["phases"])
        full_scene = np.concatenate(
            (x["retina_intensity"][:, :, None], x["motion_activity"]), axis=2
        )
        body_values, body_valid = y["values"].copy(), y["valid"].copy()
    complete_history(times, body_times)
    if (
        full_scene.shape != (len(times), 4, 9, 721)
        or body_values.shape != (len(times), 21)
        or body_valid.shape != body_values.shape
        or body_valid.dtype != bool
        or not np.isfinite(full_scene).all()
        or not np.isfinite(body_values).all()
        or not body_valid[:, [0, 1, 4, 16]].all()
    ):
        raise ValueError("Complete real four-view/body data and original masks required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "training_result_sha256": sha256(training / "result.json"),
            "candidate_sha256": cr["checkpoint_sha256"],
            "candidate_result_sha256": sha256(candidate / "result.json"),
            "sensors_result_sha256": sha256(sensors / "result.json"),
            "labels_dataset_sha256": sha256(labels / "dataset.json"),
            "vision_sha256": vr["responses_sha256"],
            "body_sha256": br["virtual_body_sha256"],
            "recording_events_sha256": vm["events_sha256"],
            "teacher_receipt_sha256": sha256(teacher_receipt),
            "label_sha256": label["targets_sha256"],
            "recorded_camera_origins_m": origins.tolist(),
            "goal_kmh": 30,
            "recorded_teacher_goal_kmh": 60,
            "goal_is_counterfactual": True,
            "direct_motion_mean_is_intervention": True,
            "target_used_only_after_inference": True,
            "known_development_not_blind": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    n = len(times)
    # This explicit 30 km/h actor goal is a counterfactual: source AI was instructed 60 km/h.
    goals = speed_goals(30, n)
    visual = torch.zeros((n, 6), dtype=torch.float32, device="cuda")
    visual_valid = torch.zeros((n, 6), dtype=torch.bool, device="cuda")
    with torch.no_grad():
        scene = model.backbone.scene_pool(
            torch.tensor(full_scene, dtype=torch.float32, device="cuda")
        )
        body_t = torch.tensor(body_values, dtype=torch.float32, device="cuda")
        valid_t = torch.tensor(body_valid, device="cuda")
        goal_t = torch.tensor(goals, device="cuda")
        mean_scene = scene.clone()
        motion = torch.zeros(576, dtype=torch.bool, device="cuda").reshape(4, 9, 16)
        motion[:, 1:] = True
        motion = motion.flatten()
        mean_scene[:, motion] = model.backbone.scene_command.mean[motion]
        drives = []
        direct = torch.cat((model.backbone.steering_ports, model.backbone.goal_ports))
        for weight, bias in (
            (orig_weight, orig_bias),
            (delta["weight"].to("cuda"), delta["bias"].to("cuda")),
        ):
            model.backbone.vision.weight.copy_(weight)
            model.backbone.vision.bias.copy_(bias)
            full = model._drive_tensor(visual, visual_valid, body_t, valid_t, goal_t, scene)
            mean = model._drive_tensor(visual, visual_valid, body_t, valid_t, goal_t, mean_scene)
            direct_clamped = full.clone()
            direct_clamped[:, direct] = mean[:, direct]
            drives.extend((full, direct_clamped))
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(times)]
        state = model.initial_state(4)
        held = None
        output = []
        for i, intervals in enumerate(schedule):
            if i:
                for dt in intervals:
                    state = model._advance_elapsed_tensor(held, state, dt)
            output.append(model._read_state_tensor(state)[0].cpu().numpy())
            held = torch.cat([drive[i : i + 1] for drive in drives])
    output = np.stack(output)
    # Labels are first opened only after inference is complete.
    with load_array(label["targets_file"], allow_pickle=False) as x:
        np.testing.assert_array_equal(times, x["observation_time"])
        preview, valid = x["preview"].copy(), x["preview_valid"].copy()
    if preview.shape != (n, 3, 2) or valid.shape != (n, 3) or valid.dtype != bool:
        raise ValueError("Original time-aligned privileged diagnostic targets required")
    curvature = preview[:, 2, 1]
    active = valid[:, 2] & (np.abs(curvature) > 0.001)
    result = {}
    names = (
        "original_full",
        "original_direct_motion_mean",
        "candidate_full",
        "candidate_direct_motion_mean",
    )
    for k, name in enumerate(names):
        steering = output[:, k, 0]
        expected = np.sign(steering[active]) == -np.sign(curvature[active])
        result[name] = {
            "positive_steer_count": int((steering > 0).sum()),
            "negative_steer_count": int((steering < 0).sum()),
            "target_sign_matches": int(expected.sum()),
            "target_sign_count": int(active.sum()),
            "mean_abs_steering": float(np.abs(steering).mean()),
        }
    np.savez_compressed(
        run / "predictions.npz",
        output=output,
        time=times,
        target_curvature=curvature,
        target_valid=valid[:, 2],
        active=active,
    )
    write_json(
        run / "result.json",
        {
            "status": "completed_exterior_fixed_flight_replay",
            "conditions": result,
            "target_positive_count": int((curvature[active] > 0).sum()),
            "target_negative_count": int((curvature[active] < 0).sum()),
            "predictions_sha256": sha256(run / "predictions.npz"),
            "camera_mount_differs_from_training": True,
            "goal_mismatch_counterfactual": True,
            "label_is_teacher_path_not_optimal_line": True,
            "known_development_not_independent": True,
            "physical_exposure_sync_verified": False,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )
    print(
        json.dumps(
            {
                "conditions": result,
                "target_positive": int((curvature[active] > 0).sum()),
                "target_negative": int((curvature[active] < 0).sum()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
