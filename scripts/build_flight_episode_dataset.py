"""Seal separate complete flight episodes with explicit speed tasks; no driving or training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_learning import (
    GOAL_CONTRACT,
    complete_history,
    signed_intent_targets,
    speed_goals,
    teacher_speed,
)
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def original_history(root):
    clocks, phases = [], []
    for row in (json.loads(line) for line in (root / "events.jsonl").open(encoding="utf8")):
        kind = row.get("kind")
        if kind in ("student_request", "handoff_wait_observation"):
            clocks.append(row["actor_input"]["decision_monotonic_seconds"])
            phases.append(kind)
        elif kind == "expert_observation":
            clocks.append(row["decision_monotonic_seconds"])
            phases.append(kind)
        elif kind is None and all(key in row for key in ("native", "camera",
                                                        "decision_monotonic_seconds")):
            clocks.append(row["decision_monotonic_seconds"])
            phases.append("engine_ai_observation")
    return np.array(clocks), np.array(phases)


def build(entry, spec, pool, run, number):
    root, visual, body = (Path(entry[key]) for key in ("recording", "scene_replay", "body_replay"))
    source_hash = sha256(root / "events.jsonl")
    clocks, phases = original_history(root)
    identity = read(root / "identity.json")
    body_contract = read("configs/ac-virtual-body-v1.json")
    if any(identity.get(k) != v for k, v in body_contract["native_identity"].items()):
        raise ValueError("Original native identity differs from registered body contract")
    visual_receipt, body_receipt = read(visual / "result.json"), read(body / "result.json")
    for replay in (visual, body):
        if read(replay / "manifest.json")["config"]["events_sha256"] != source_hash:
            raise ValueError("Replay refers to a different original history")
    if (visual_receipt["status"] != "completed" or body_receipt["status"] != "completed"
            or sha256(visual / "responses.npz") != visual_receipt["responses_sha256"]
            or sha256(body / "virtual-body.npz") != body_receipt["virtual_body_sha256"]
            or body_receipt["contract_sha256"] != sha256("configs/ac-virtual-body-v1.json")):
        raise ValueError("Unchanged complete successful visual/body replays required")
    with (load_array(spec["scene_registration_file"], allow_pickle=False) as expected,
          load_array(visual / "retina-registration.npz", allow_pickle=False) as current):
        for key in ("image_uv", "body_directions", "view_order", "motion_types"):
            np.testing.assert_array_equal(expected[key], current[key])
    with load_array(visual / "responses.npz", allow_pickle=False) as archive:
        complete_history(archive["decision_monotonic_seconds"], clocks)
        np.testing.assert_array_equal(archive["phases"], phases)
        scene = np.concatenate([archive["retina_intensity"][:, :, None],
                                archive["motion_activity"]], axis=2)
    if scene.shape != (len(clocks), 4, 9, 721) or not np.isfinite(scene).all():
        raise ValueError("Every original retinotopic four-camera scene required")
    with torch.no_grad():
        features = pool(torch.tensor(scene, dtype=torch.float32, device="cuda")).cpu().numpy()
    with load_array(body / "virtual-body.npz", allow_pickle=False) as archive:
        complete_history(archive["decision_monotonic_seconds"], clocks)
        np.testing.assert_array_equal(archive["phases"], phases)
        values, valid_body = archive["values"].copy(), archive["valid"].copy()
    if values.shape != (len(clocks), 16) or not valid_body[:, [0, 1, 4]].all():
        raise ValueError("Registered body tilt/yaw required for every original observation")
    teacher_root = Path(entry["teacher_dataset"])
    manifest = read(teacher_root / "dataset.json")
    speed = teacher_speed(manifest["teacher_label_contract"])
    if speed != entry["desired_speed_kmh"]:
        raise ValueError("Declared actor task differs from original teacher speed")
    selected = [e for e in manifest["episodes"] if e["file"] == entry["teacher_episode"]]
    if len(selected) != 1:
        raise ValueError("One explicit original teacher episode required")
    episode = selected[0]
    original = episode.get("original_recording") or episode.get("original_trial")
    if original is None or Path(original).resolve() != root.resolve():
        raise ValueError("Teacher episode refers to a different original recording")
    teacher_file = resolve_episode_path(teacher_root, episode, manifest)
    if sha256(teacher_file) != episode["sha256"]:
        raise ValueError("Original teacher labels changed")
    with load_array(teacher_file, allow_pickle=False) as archive:
        complete_history(archive["observation_time"], clocks)
        actions, valid = archive["action"].copy(), archive["action_valid"].copy()
    target, mask = signed_intent_targets(actions, valid, phases)
    if not mask.any(axis=0).all():
        raise ValueError("At least one original valid target per intent required")
    file = run / f"episode-{number:02d}.npz"
    np.savez_compressed(file, scene_features=features, body=values.astype(np.float32),
                        body_valid=valid_body, visual=np.zeros((len(clocks), 6), np.float32),
                        visual_valid=np.zeros((len(clocks), 6), bool),
                        goals=speed_goals(speed, len(clocks)), target=target, target_valid=mask,
                        actions=actions, action_valid=valid, observation_time=clocks, phases=phases)
    return dict(file=file.name, sha256=sha256(file), role=entry["role"],
                observations=len(clocks), desired_speed_kmh=speed, original=entry,
                source_events_sha256=source_hash, source_teacher_sha256=episode["sha256"],
                source_visual_sha256=visual_receipt["responses_sha256"],
                source_body_sha256=body_receipt["virtual_body_sha256"],
                teacher_label_contract=manifest["teacher_label_contract"],
                valid_targets=mask.sum(0).tolist(),
                left_labels=int(((actions[:, 0] < -.01) & valid[:, 0]).sum()),
                right_labels=int(((actions[:, 0] > .01) & valid[:, 0]).sum()),
                brake_labels_above01=int(((actions[:, 2] > .01) & valid[:, 2]).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    spec = read(plan["spec"])
    if spec.get("goal_contract") != GOAL_CONTRACT:
        raise ValueError("New explicit speed-goal specification required; no v3 reinterpretation")
    if sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]:
        raise ValueError("Registered visual geometry changed")
    roles = {e["role"] for e in plan["episodes"]}
    if roles != {"train", "development"}:
        raise ValueError("Separate declared training/development recordings required")
    hashes = [sha256(Path(e["recording"]) / "events.jsonl") for e in plan["episodes"]]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Same original history cannot be duplicated or split between roles")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU feature cache required; no CPU fallback")
    torch.set_num_threads(1)
    with load_array(spec["scene_registration_file"], allow_pickle=False) as registration:
        pool = FlightScenePool(registration["image_uv"], spec["scene_basis"]).to("cuda")
    run = begin_run(args.output, dict(arguments=vars(args), seed=plan["seed"],
                                    plan_sha256=sha256(args.plan), spec_sha256=sha256(plan["spec"]),
                                    actuator_messages_sent=0, optimizer_updates=0))
    records = []
    for number, entry in enumerate(plan["episodes"]):
        records.append(build(entry, spec, pool, run, number))
        write_json(run / "progress.json", dict(completed=len(records), total=len(plan["episodes"])))
    manifest = dict(schema="flybrain.flight.episodes.v1", goal_contract=GOAL_CONTRACT,
                    spec=plan["spec"], spec_sha256=sha256(plan["spec"]), episodes=records,
                    body_contract_sha256=sha256("configs/ac-virtual-body-v1.json"),
                    no_temporal_slicing=True, no_reset_at_handoff=True,
                    development_is_blind=False, actor_qualified=False)
    write_json(run / "dataset.json", manifest)
    write_json(run / "result.json", dict(status="completed", episodes=len(records),
               termination_reason="complete histories sealed with distinct task and episode roles",
               dataset_manifest_sha256=sha256(run / "dataset.json"), optimizer_updates=0,
               actuator_messages_sent=0, goal_achieved=False))


if __name__ == "__main__":
    main()
