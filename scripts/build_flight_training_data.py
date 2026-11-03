"""Seal aligned frozen visual responses, native body feedback and original teacher masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/flight-initial-training-20261029.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
    spec = json.loads(Path(plan["spec"]).read_text(encoding="utf8"))
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            plan=plan,
            seed=plan["seed"],
            plan_sha256=sha256(args.plan),
            spec_sha256=sha256(plan["spec"]),
            actuator_messages_sent=0,
        ),
    )
    root = Path(plan["scene_replay"])
    scene_receipt = json.loads((root / "result.json").read_text(encoding="utf8"))
    if sha256(root / "responses.npz") != scene_receipt["responses_sha256"]:
        raise ValueError("Frozen visual response changed")
    with load_array(root / "responses.npz", allow_pickle=False) as archive:
        retinal = archive["retina_intensity"].copy()
        motion = archive["motion_activity"].copy()
        clocks = archive["decision_monotonic_seconds"].copy()
        phases = archive["phases"].copy()
    body_root = Path(plan["body_replay"])
    body_receipt = json.loads((body_root / "result.json").read_text(encoding="utf8"))
    if sha256(body_root / "virtual-body.npz") != body_receipt["virtual_body_sha256"]:
        raise ValueError("Native virtual-body replay changed")
    with load_array(body_root / "virtual-body.npz", allow_pickle=False) as archive:
        body, body_valid = archive["values"].copy(), archive["valid"].copy()
        np.testing.assert_array_equal(clocks, archive["decision_monotonic_seconds"])
        np.testing.assert_array_equal(phases, archive["phases"])
    teacher = Path(plan["teacher_dataset"])
    manifest = json.loads((teacher / "dataset.json").read_text(encoding="utf8"))
    episode = manifest["episodes"][0]
    teacher_file = resolve_episode_path(teacher, episode, manifest)
    if len(manifest["episodes"]) != 1 or sha256(teacher_file) != episode["sha256"]:
        raise ValueError("Original single sealed episode changed")
    with load_array(teacher_file, allow_pickle=False) as archive:
        actions, valid = archive["action"].copy(), archive["action_valid"].copy()
        np.testing.assert_array_equal(clocks, archive["observation_time"])
    if len(clocks) != 1543 or not np.isfinite(clocks).all() or np.any(np.diff(clocks) <= 0):
        raise ValueError("Entire original increasing decision history required")
    if valid[:807].any() or not np.array_equal(
        np.flatnonzero(valid.all(axis=1)), np.arange(815, 1537)
    ):
        raise ValueError("Student/wait masks or original teacher validity changed")
    raw_rows = []
    for row in (
        json.loads(line)
        for line in (Path(plan["raw_recording"]) / "events.jsonl").open(encoding="utf8")
    ):
        if row["kind"] == "expert_raw_native":
            native = row["native"]["fields"]
            if native["speedKmh"] > 8:
                raw_rows.append([native["steerAngle"], native["localAngularVel"][1]])
    raw = np.array(raw_rows)
    correlation = float(np.corrcoef(raw.T)[0, 1])
    if len(raw) < 100 or correlation >= -0.8:
        raise ValueError("Negative native steering/yaw sign correspondence not confirmed")
    if sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]:
        raise ValueError("Retinal registration changed")
    with load_array(spec["scene_registration_file"], allow_pickle=False) as archive:
        pool = FlightScenePool(archive["image_uv"], spec["scene_basis"])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for deterministic frozen feature cache")
    pool = pool.to("cuda")
    with torch.no_grad():
        scene = np.concatenate([retinal[:, :, None], motion], axis=2)
        tensor = torch.tensor(scene, device="cuda", dtype=torch.float32)
        features = pool(tensor).cpu().numpy()
        torch.testing.assert_close(pool(tensor[:2]), torch.tensor(features[:2], device="cuda"))
    overlap = valid[:, 1] & valid[:, 2] & (actions[:, 1] > 0) & (actions[:, 2] > 0)
    if overlap.any():
        raise ValueError('This initial signed-drive codec cannot represent simultaneous pedals')
    target = np.column_stack([-actions[:, 0], actions[:, 1] - actions[:, 2]])
    target_valid = np.column_stack([valid[:, 0], valid[:, 1] & valid[:, 2]])
    indices = np.arange(len(clocks))
    train = (indices >= plan["train_label_observations"][0]) & (
        indices < plan["train_label_observations"][1]
    )
    development = (indices >= plan["validation_label_observations"][0]) & (
        indices < plan["validation_label_observations"][1]
    )
    np.savez_compressed(
        run / "episode.npz",
        scene_features=features,
        body=body.astype(np.float32),
        body_valid=body_valid,
        visual=np.zeros((len(clocks), 6), np.float32),
        visual_valid=np.zeros((len(clocks), 6), bool),
        goals=np.tile(np.array(plan["task_goal"], np.float32), (len(clocks), 1)),
        target=target.astype(np.float32),
        target_valid=target_valid,
        actions=actions,
        action_valid=valid,
        train_mask=train,
        development_mask=development,
        observation_time=clocks,
        phases=phases,
    )
    write_json(
        run / "result.json",
        dict(
            status="completed",
            termination_reason="all original observations sealed with new spatial/body contract",
            observations=len(clocks),
            scene_features=features.shape[1],
            training_label_counts=(target_valid & train[:, None]).sum(axis=0).tolist(),
            development_label_counts=(target_valid & development[:, None]).sum(axis=0).tolist(),
            steer_target_mean=actions[train & valid[:, 0], 0].mean().item(),
            steer_target_std=actions[train & valid[:, 0], 0].std().item(),
            diagnostic_native_steering_yaw_correlation=correlation,
            positive_brake_labels=int((actions[valid[:, 2], 2] > 0).sum()),
            entire_history_and_clocks_preserved=True,
            source_vision_sha256=scene_receipt["responses_sha256"],
            source_body_sha256=body_receipt["virtual_body_sha256"],
            source_teacher_sha256=episode["sha256"],
            body_contract_sha256=body_receipt["contract_sha256"],
            plan_sha256=sha256(args.plan),
            actor_world_pose_or_teacher_input=False,
            scene_motion_estimate_unregistered=True,
            dataset_sha256=sha256(run / "episode.npz"),
            actuator_messages_sent=0,
            goal_achieved=False,
        ),
    )
    print(
        json.dumps(
            dict(
                observations=len(clocks), train=int(train.sum()), development=int(development.sum())
            )
        )
    )


if __name__ == "__main__":
    main()
