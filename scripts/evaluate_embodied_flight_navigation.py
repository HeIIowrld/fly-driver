"""Fixed development comparison and information removal after terminal joint training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def metrics(prediction, target, valid):
    heads = []
    for head, name in enumerate(("left_turn", "forward")):
        p, y = prediction[valid[:, head], head], target[valid[:, head], head]
        heads.append(
            dict(
                head=name,
                count=len(y),
                rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                prediction_mean=float(p.mean()),
                target_mean=float(y.mean()),
            )
        )
    return heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training = Path(args.training)
    receipt = read(training / "result.json")
    assert receipt["status"] == "completed_fixed_joint_command_training"
    plan = read(training / "manifest.json")["config"]["plan"]
    assert len(receipt["epochs"]) == plan["epochs"] == 24
    assert sha256(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
    source = read(Path(plan["source"]) / "result.json")
    assert sha256(source["checkpoint"]) == source["checkpoint_sha256"]
    root = Path(plan["data"])
    dataset = read(root / "dataset.json")
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    assert payload["spec"] == model.spec and payload["schema"] == model.schema
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    entries, training_targets, training_masks, training_vehicle = [], [], [], []
    for entry in dataset["episodes"]:
        assert entry["role"] in ("train", "development")
        assert sha256(root / entry["file"]) == entry["sha256"]
        with load_array(root / entry["file"], allow_pickle=False) as archive:
            a = {k: archive[k].copy() for k in archive.files}
        assert not a["visual_valid"].any(), (
            "Removal below assumes original functional visual6 invalid"
        )
        entries.append((entry, a))
        if entry["role"] == "train":
            training_targets.append(a["target"])
            training_masks.append(a["target_valid"])
            training_vehicle.append(a["body"][:, 16:])
    train_y, train_valid = np.concatenate(training_targets), np.concatenate(training_masks)
    mean = np.array([train_y[train_valid[:, h], h].mean() for h in range(2)])
    vehicle_mean = np.concatenate(training_vehicle).mean(0)
    conditions = (
        "intact",
        "new_command_removed",
        "all_visual_train_mean",
        "motion_train_mean",
        "intensity_train_mean",
        "vehicle_feedback_train_mean",
    )
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            program_sha256=sha256(__file__),
            training_result_sha256=sha256(training / "result.json"),
            checkpoint_sha256=receipt["checkpoint_sha256"],
            conditions=conditions,
            checkpoint_selection="Fixed last24 epochs, no development selection",
            known_development_not_blind=True,
            evaluation_arrays_opened=False,
            interpretation="Input-information replacement is not a physiological neuron lesion",
        ),
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    reports, saved, gates = [], {}, []
    with torch.no_grad():
        for entry, a in entries:
            if entry["role"] != "development":
                continue
            tensor = {
                k: torch.tensor(a[k], device="cuda")
                for k in ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
            }
            schedule = [[]] + [substeps(float(t)) for t in np.diff(a["observation_time"])]
            result = dict(
                file=entry["file"],
                conditions={},
                train_mean=metrics(
                    np.tile(mean, (len(a["target"]), 1)), a["target"], a["target_valid"]
                ),
                zero=metrics(np.zeros_like(a["target"]), a["target"], a["target_valid"]),
            )
            for condition in conditions:
                scene, body = tensor["scene_features"].clone(), tensor["body"].clone()
                if condition in (
                    "all_visual_train_mean",
                    "motion_train_mean",
                    "intensity_train_mean",
                ):
                    selected = torch.zeros(576, device="cuda", dtype=torch.bool).reshape(4, 9, 16)
                    if condition == "all_visual_train_mean":
                        selected[:] = True
                    elif condition == "motion_train_mean":
                        selected[:, 1:] = True
                    else:
                        selected[:, :1] = True
                    selected = selected.flatten()
                    scene[:, selected] = model.backbone.scene_command.mean[selected]
                if condition == "vehicle_feedback_train_mean":
                    body[:, 16:] = torch.tensor(vehicle_mean, device="cuda")
                drive = model._drive_tensor(
                    tensor["visual"],
                    tensor["visual_valid"],
                    body,
                    tensor["body_valid"],
                    tensor["goals"],
                    scene,
                    lesion="new_command" if condition == "new_command_removed" else None,
                )
                state, held = model.initial_state(1), None
                predictions = []
                for index in range(len(schedule)):
                    if index:
                        for dt in schedule[index]:
                            state = model._advance_elapsed_tensor(held, state, dt)
                    predictions.append(model._read_state_tensor(state)[0])
                    held = drive[index : index + 1]
                prediction = torch.cat(predictions, 0).cpu().numpy()
                assert np.isfinite(prediction).all()
                late = (
                    a["target_valid"]
                    & (a["observation_time"] - a["observation_time"][0] >= 3)[:, None]
                )
                result["conditions"][condition] = dict(
                    full=metrics(prediction, a["target"], a["target_valid"]),
                    after3s=metrics(prediction, a["target"], late),
                    negative_forward_predictions_without_brake_supervision=int(
                        (prediction[:, 1] < -0.01).sum()
                    ),
                )
                saved[f"{entry['file'].removesuffix('.npz')}_{condition}"] = prediction
            intact = result["conditions"]["intact"]["full"]
            removed = result["conditions"]["all_visual_train_mean"]["full"]
            motion = result["conditions"]["motion_train_mean"]["full"]
            source_metrics = result["conditions"]["new_command_removed"]["full"]
            feedback = result["conditions"]["vehicle_feedback_train_mean"]["full"]
            gates.append(
                dict(
                    file=entry["file"],
                    steering_better_than_mean_and_source=intact[0]["rmse"]
                    < min(result["train_mean"][0]["rmse"], source_metrics[0]["rmse"]),
                    visual_information_helps_steering=intact[0]["rmse"] + 1e-6 < removed[0]["rmse"],
                    motion_information_helps_steering=intact[0]["rmse"] + 1e-6 < motion[0]["rmse"],
                    forward_better_than_mean_and_source=intact[1]["rmse"]
                    < min(result["train_mean"][1]["rmse"], source_metrics[1]["rmse"]),
                    vehicle_feedback_helps_forward=intact[1]["rmse"] + 1e-6 < feedback[1]["rmse"],
                )
            )
            reports.append(result)
            write_json(
                run / "progress.json",
                dict(completed_development_episodes=len(reports), gates=gates),
            )
    np.savez_compressed(run / "development-predictions.npz", **saved)
    assert sha256(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
    write_json(
        run / "result.json",
        dict(
            status="completed_fixed_embodied_development_evaluation",
            reports=reports,
            gates=gates,
            complete_learning_gate_passed=bool(gates)
            and all(all(value for key, value in row.items() if key != "file") for row in gates),
            physiological_or_closed_loop_qualification=False,
            braking_verified=False,
            evaluation_role_arrays_opened=False,
            optimizer_updates=0,
            actuator_messages_sent=0,
            ui_actions_sent=0,
            goal_achieved=False,
        ),
    )
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
