"""Fixed existing flight candidate on a newly audited aid-OFF complete driving recording."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.audit_flight_speed_training import partitioned
from scripts.build_flight_episode_dataset import build
from scripts.train_flight_initial import substeps
from scripts.train_flight_visual_boundary import evaluate_visual


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    training, entry = Path(plan["training"]), plan["episode"]
    source = Path(entry["recording"])
    receipt, config = read(training / "result.json"), read(training / "manifest.json")["config"]
    raw, raw_config = read(Path(plan["raw_audit"]) / "result.json"), read(
        Path(plan["raw_audit"]) / "manifest.json")["config"]
    teacher = read(Path(entry["teacher_dataset"]) / "dataset.json")
    teacher_receipt = read(Path(entry["teacher_dataset"]) / "result.json")
    vision_audit = read(Path(plan["vision_audit"]) / "result.json")
    vision_audit_config = read(Path(plan["vision_audit"]) / "manifest.json")["config"]
    if (sha256(receipt["checkpoint"]) != plan["checkpoint_sha256"]
            or receipt["checkpoint_sha256"] != plan["checkpoint_sha256"]
            or raw["status"] != "completed" or raw.get("camera_aid_state_verified") is not True
            or raw["intervals_above_80ms"] or any(raw["actor_replay_max_errors"].values())
            or raw_config["events_sha256"] != sha256(source / "events.jsonl")
            or raw_config["observation_sha256"] != sha256(source / "observations.npz")
            or raw_config["bundle_sha256"] != sha256(source / "vision/camera.zip")
            or teacher.get("camera_aid_state_verified") is not True
            or sha256(Path(entry["teacher_dataset"]) / "dataset.json")
            != teacher_receipt["dataset_manifest_sha256"]
            or vision_audit["status"] != "completed"
            or vision_audit_config["input_result_sha256"]
            != sha256(Path(entry["scene_replay"]) / "result.json")):
        raise ValueError("Fixed candidate and fully audited original aid-OFF inputs required")
    visual_spec = config["plan"]["spec"]
    if sha256(visual_spec) != config["spec_sha256"]:
        raise ValueError("Original candidate spec changed")
    data_spec = read(plan["data_spec"])
    for key in ("scene_registration_sha256", "scene_basis"):
        if data_spec[key] != read(visual_spec)[key]:
            raise ValueError("Feature extraction changed from original candidate")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan, seed=plan["seed"],
        plan_sha256=sha256(args.plan), checkpoint_sha256=plan["checkpoint_sha256"],
        raw_audit_manifest_sha256=sha256(Path(plan["raw_audit"]) / "manifest.json"),
        vision_audit_manifest_sha256=sha256(Path(plan["vision_audit"]) / "manifest.json"),
        data_spec_sha256=sha256(plan["data_spec"]), normalization_recomputed=False,
        optimizer_updates=0, actuator_messages_sent=0))
    with load_array(data_spec["scene_registration_file"], allow_pickle=False) as registration:
        pool = FlightScenePool(registration["image_uv"], data_spec["scene_basis"]).to("cuda")
    record = build(entry, data_spec, pool, run, 0)
    with load_array(run / record["file"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
            if value.dtype.kind in "fbiu" and key != "observation_time"}
    schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
    episodes = [(record, data, schedule, arrays)]
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    model = VisualFlightCircuit(visual_spec, glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    model.assert_body_boundary_protected()
    metrics, predictions = evaluate_visual(model, episodes)
    split = partitioned(model, data, schedule)
    np.testing.assert_allclose(split, predictions[0], atol=3e-6, rtol=2e-5)
    lesions = {}
    for name in ("scene_command", "vision", "motion_channels", "intensity_channel"):
        reports, values = evaluate_visual(model, episodes, lesion=name)
        lesions[name] = reports[0]
        np.save(run / f"prediction-{name}.npy", values[0])
    old_root = Path(config["plan"]["data"])
    if sha256(old_root / "dataset.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Original train-mean reference manifest changed")
    train_targets = []
    for old in read(old_root / "dataset.json")["episodes"]:
        if old["role"] == "train" and old["desired_speed_kmh"] == record["desired_speed_kmh"]:
            if sha256(old_root / old["file"]) != old["sha256"]:
                raise ValueError("Original training reference changed")
            with load_array(old_root / old["file"], allow_pickle=False) as archive:
                valid, target = archive["target_valid"], archive["target"]
                train_targets.append(target[valid[:, 0], 0])
    mean = float(np.concatenate(train_targets).mean())
    valid = arrays["target_valid"][:, 0]
    time = arrays["observation_time"] - arrays["observation_time"][0]
    baseline, flags = {}, {}
    for name, selected in [("full", valid), ("late", valid & (time >= 3.))]:
        score = metrics[0]["rmse"][0] if name == "full" else metrics[0]["late_turn_rmse"]
        baseline[name] = float(np.mean((arrays["target"][selected, 0] - mean) ** 2) ** .5)
        flags[f"beats_mean_{name}"] = score < baseline[name]
        for lesion in ("scene_command", "vision"):
            other = lesions[lesion]["rmse"][0] if name == "full" else lesions[lesion][
                "late_turn_rmse"]
            flags[f"beats_{lesion}_{name}"] = score < other
    normalized = (data["scene_features"] - model.scene_command.mean) / model.scene_command.scale
    with torch.no_grad():
        currents = model.scene_command(data["scene_features"])
    np.save(run / "prediction.npy", predictions[0])
    import matplotlib.pyplot as plt

    figure, panel = plt.subplots(figsize=(11, 4), layout="constrained")
    panel.plot(time, np.where(valid, arrays["target"][:, 0], np.nan),
               label="Applied-control target")
    panel.plot(time, predictions[0][:, 0], label="Frozen guarded v6")
    panel.plot(time, load_array(run / "prediction-scene_command.npy")[:, 0],
               label="New scene information removed")
    panel.set(xlabel="Original elapsed time (s)", ylabel="Left-turn intent",
              title="Aid-OFF recording / fixed candidate / no closed-loop driving claim")
    panel.legend()
    figure.savefig(run / "fresh-aid-off-fixed-fit.png", dpi=130)
    plt.close(figure)
    assert sha256(receipt["checkpoint"]) == plan["checkpoint_sha256"]
    write_json(run / "result.json", dict(status="completed_fixed_fresh_recording_evaluation",
        source_record=record, metrics=metrics[0], input_lesions=lesions,
        original_training_speed_mean=mean, speed_mean_baseline=baseline, checks=flags,
        visual_gate_passed=all(flags.values()), camera_aid_state_verified=True,
        normalized_feature_clip_fraction=float((normalized.abs() >= 5).float().mean()),
        scene_current_min=currents.min(0).values.cpu().tolist(),
        scene_current_max=currents.max(0).values.cpu().tolist(),
        chunk37_error=float(np.abs(split - predictions[0]).max()),
        normalization_recomputed=False, checkpoint_modified=False, original_clocks_preserved=True,
        evaluation_is_blind=False, known_recording_reuse=plan.get("known_recording_reuse", False),
        optimizer_updates=0, actor_qualified=False, physiological_recognition_verified=False,
        physical_vehicle_plant_tested=False, actual_actor_integration=False,
        actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(status="completed", visual_gate=all(flags.values()))))


if __name__ == "__main__":
    main()
