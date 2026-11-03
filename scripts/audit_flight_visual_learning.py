"""Audit v6 frozen state, train-only moments, causal replay and visual controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.audit_flight_speed_training import partitioned
from scripts.train_flight_initial import replay, substeps
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
    parent_receipt = read(Path(plan["source"]) / "result.json")
    if (
        receipt["status"] != "completed_offline_visual_experiment"
        or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
        or sha256(parent_receipt["checkpoint"]) != config["source_checkpoint_sha256"]
        or sha256(plan["spec"]) != config["spec_sha256"]
        or sha256(data_root / "dataset.json") != config["dataset_manifest_sha256"]
    ):
        raise ValueError("Successful unchanged fit, source and data required")
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=863151,
            checkpoint_sha256=receipt["checkpoint_sha256"],
            source_config_sha256=sha256(root / "manifest.json"),
            partition_observations=37,
            camera_negative_control=[3, 2, 1, 0],
            camera_control_scope="wrong view registration, model-only diagnostic, not actor data",
            actuator_messages_sent=0,
            optimizer_updates=0,
        ),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    parent = torch.load(parent_receipt["checkpoint"], map_location="cpu", weights_only=True)
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    for key, value in parent["model"].items():
        current = payload["model"][key]
        if value.is_sparse:
            if not torch.equal(
                value.coalesce().indices(), current.coalesce().indices()
            ) or not torch.equal(value.coalesce().values(), current.coalesce().values()):
                raise ValueError("Inherited graph changed")
        elif not torch.equal(value, current):
            raise ValueError("Inherited parameter/buffer changed: " + key)
    model = VisualFlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.assert_body_boundary_protected()
    episodes, matrices = [], []
    for entry in read(data_root / "dataset.json")["episodes"]:
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Source original episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if entry["role"] == "train":
            matrices.append(arrays["scene_features"].astype(np.float64))
        data = {
            key: torch.tensor(value, device="cuda")
            for key, value in arrays.items()
            if value.dtype.kind in "fbiu" and key != "observation_time"
        }
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        episodes.append((entry, data, schedule, arrays))
    features = np.concatenate(matrices)
    mean, scale = features.mean(0), features.std(0).clip(min=plan["normalization_std_floor"])
    saved_mean, saved_scale = (
        model.scene_command.mean.cpu().numpy(),
        model.scene_command.scale.cpu().numpy(),
    )
    np.testing.assert_allclose(saved_mean, mean, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(saved_scale, scale, atol=1e-6, rtol=1e-5)
    reports = []
    for entry, data, schedule, arrays in episodes:
        prediction = replay(model, data, schedule)[0].cpu().numpy()
        split = partitioned(model, data, schedule)
        np.testing.assert_allclose(prediction, split, atol=3e-6, rtol=2e-5)
        with load_array(root / entry["file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["observation_time"], arrays["observation_time"])
            np.testing.assert_array_equal(saved["target_valid"], arrays["target_valid"])
            np.testing.assert_allclose(saved["prediction"], prediction, atol=3e-6, rtol=2e-5)
            error = float(np.abs(saved["prediction"] - prediction).max())
        selected = arrays["target_valid"][:, 0]
        late = selected & (arrays["observation_time"] - arrays["observation_time"][0] >= 3)
        reports.append(
            dict(
                file=entry["file"],
                role=entry["role"],
                observations=len(schedule),
                saved_prediction_error=error,
                chunk37_error=float(np.abs(split - prediction).max()),
                zero_turn_rmse=float(np.sqrt(np.mean(arrays["target"][selected, 0] ** 2))),
                late_zero_turn_rmse=float(np.sqrt(np.mean(arrays["target"][late, 0] ** 2))),
            )
        )
        write_json(run / "progress.json", dict(completed=len(reports), total=5))
    corrupted = []
    for entry, data, schedule, arrays in episodes:
        wrong = dict(data)
        wrong["scene_features"] = (
            data["scene_features"].reshape(-1, 4, 9, 16)[:, [3, 2, 1, 0]].flatten(1)
        )
        corrupted.append((entry, wrong, schedule, arrays))
    camera_metrics, _ = evaluate_visual(model, corrupted)
    removed_metrics, removed_predictions = evaluate_visual(model, episodes, lesion="scene_command")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 2, figsize=(13, 14), layout="constrained")
    for number, ((entry, _, _, arrays), removed) in enumerate(
        zip(episodes, removed_predictions, strict=True)
    ):
        with load_array(root / entry["file"], allow_pickle=False) as saved:
            prediction = saved["prediction"][:, 0]
        time = arrays["observation_time"] - arrays["observation_time"][0]
        target = np.where(arrays["target_valid"][:, 0], arrays["target"][:, 0], np.nan)
        for column, late in enumerate([False, True]):
            selected = time >= 3 if late else np.ones(len(time), bool)
            panel = axes[number, column]
            panel.plot(
                time[selected], target[selected], label="Applied-control target", linewidth=0.8
            )
            panel.plot(
                time[selected], prediction[selected], label="Visual candidate", linewidth=0.8
            )
            panel.plot(
                time[selected], removed[selected, 0], label="New scene info removed", linewidth=0.7
            )
            panel.set(
                xlabel="Original elapsed time (s)",
                ylabel="Left-turn intent",
                title=f"{entry['file']} {entry['role']} / {'after3s' if late else 'full history'}",
            )
            panel.legend(fontsize=6)
    figure.savefig(run / "visual-fit-information-controls.png", dpi=140)
    plt.close(figure)
    write_json(
        run / "result.json",
        dict(
            status="completed",
            episodes=reports,
            independent_normalization_mean_error=float(np.abs(saved_mean - mean).max()),
            independent_normalization_scale_error=float(np.abs(saved_scale - scale).max()),
            normalization_training_observations=len(features),
            normalization_uses_development=False,
            all_v5_inherited_state_unchanged=True,
            camera_registration_corruption_metrics=camera_metrics,
            scene_information_removed_metrics=removed_metrics,
            physical_vehicle_plant_tested=False,
            physiological_recognition_verified=False,
            actor_qualified=False,
            goal_achieved=False,
            actuator_messages_sent=0,
            termination_reason="strict reload and declared visual controls measured",
        ),
    )
    print(json.dumps(dict(status="completed", episodes=len(reports))))


if __name__ == "__main__":
    main()
