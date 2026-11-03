"""One fixed-checkpoint development evaluation, separate from train-only guarded fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps
from scripts.train_flight_visual_boundary import evaluate_visual


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    receipt, config = read(root / "result.json"), read(root / "manifest.json")["config"]
    plan = config["plan"]
    data_root = Path(plan["data"])
    if (receipt["status"] != "completed_guarded_optimizer_experiment"
            or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
            or sha256(plan["spec"]) != config["spec_sha256"]
            or sha256(data_root / "dataset.json") != config["dataset_manifest_sha256"]):
        raise ValueError("Fixed completed checkpoint, original data and spec required")
    run = begin_run(args.output, dict(arguments=vars(args), seed=863155,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        training_manifest_sha256=sha256(root / "manifest.json"), optimizer_updates=0,
        criterion="both development episodes must beat scene-info removal, all-vision removal "
                  "and training-speed mean on full and after3s left-turn RMSE",
        comparison_scope="known development records, not a blind course or aid-qualified data",
        checkpoint_selection="fixed prior last epoch, never select or refit from this evaluation",
        actuator_messages_sent=0))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(1)
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    model = VisualFlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    episodes, train_targets = [], {}
    for entry in read(data_root / "dataset.json")["episodes"]:
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            if entry["role"] == "train":
                target, valid = archive["target"], archive["target_valid"]
                train_targets.setdefault(entry["desired_speed_kmh"], []).append(
                    target[valid[:, 0], 0])
                continue
            if entry["role"] != "development":
                raise ValueError("Unexpected evaluation role")
            arrays = {key: archive[key].copy() for key in archive.files}
        data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                if value.dtype.kind in "fbiu" and key != "observation_time"}
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        episodes.append((entry, data, schedule, arrays))
    metrics, predictions = evaluate_visual(model, episodes)
    lesions = {name: evaluate_visual(model, episodes, lesion=name)[0]
               for name in ("scene_command", "vision", "motion_channels", "intensity_channel")}
    checks = []
    pairs = zip(episodes, predictions, strict=True)
    for i, ((entry, _, _, arrays), prediction) in enumerate(pairs):
        mean = float(np.concatenate(train_targets[entry["desired_speed_kmh"]]).mean())
        valid = arrays["target_valid"][:, 0]
        late = valid & (arrays["observation_time"] - arrays["observation_time"][0] >= 3.)
        baseline = dict(full=float(np.mean((arrays["target"][valid, 0] - mean) ** 2) ** .5),
                        late=float(np.mean((arrays["target"][late, 0] - mean) ** 2) ** .5))
        full, after = metrics[i]["rmse"][0], metrics[i]["late_turn_rmse"]
        flags = dict(
            beats_scene_full=full < lesions["scene_command"][i]["rmse"][0],
            beats_scene_late=after < lesions["scene_command"][i]["late_turn_rmse"],
            beats_vision_full=full < lesions["vision"][i]["rmse"][0],
            beats_vision_late=after < lesions["vision"][i]["late_turn_rmse"],
            beats_mean_full=full < baseline["full"], beats_mean_late=after < baseline["late"],
        )
        checks.append(dict(file=entry["file"], baseline=baseline, checks=flags,
                           all_checks_passed=all(flags.values())))
        np.savez_compressed(run / entry["file"], prediction=prediction, target=arrays["target"],
                            target_valid=arrays["target_valid"],
                            observation_time=arrays["observation_time"])
    write_json(run / "result.json", dict(status="completed_fixed_development_evaluation",
        checkpoint_sha256=receipt["checkpoint_sha256"], metrics=metrics,
        input_lesions=lesions, comparisons=checks,
        development_visual_gate_passed=all(row["all_checks_passed"] for row in checks),
        development_is_blind=False, source_recordings_aid_absence_verified=False,
        optimizer_updates=0, checkpoint_modified=False, actor_qualified=False,
        actual_actor_integration=False, physiological_recognition_verified=False,
        physical_vehicle_plant_tested=False, actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(status="completed", visual_gate=all(
        row["all_checks_passed"] for row in checks))))


if __name__ == "__main__":
    main()
