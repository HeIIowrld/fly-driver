"""Offline view interventions on frozen image-grid TRAIN-fold visual models."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.runtime)
    original = root / "20261031-unregistered-visual-fold-checkpoints"
    robust = root / "20261031-view-robust-visual-folds"
    first, second = read(original / "result.json"), read(robust / "result.json")
    if (
        first["status"] != "completed_unregistered_train_folds"
        or second["status"] != "completed_view_mean_augmented_train_folds"
    ):
        raise ValueError("Both original completed fold sets required")
    plan = read(root / "20261030-t1692-fold-setup/manifest.json")["config"]["plan"]
    dataset = Path(plan["source_data"]) / "dataset.json"
    sensors = Path(plan["sensor_replay"]) / "result.json"
    entries = {e["id"]: e for e in read(dataset)["episodes"] if e["role"] == "train"}
    sources = {e["id"]: e for e in read(sensors)["episodes"] if e["role"] == "train"}
    names = ["episode-04-train", "episode-05-train", "episode-06-train"]
    if sorted(entries) != names or sorted(sources) != names:
        raise ValueError("Three original TRAIN roles required")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "original_result_sha256": sha256(original / "result.json"),
            "robust_result_sha256": sha256(robust / "result.json"),
            "dataset_sha256": sha256(dataset),
            "sensor_result_sha256": sha256(sensors),
            "interventions": "replace all 14 channels in one view with fold TRAIN feature means",
            "output_predictions_before_targets": True,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    all_reports = []
    for name in names:
        entry, source = entries[name], sources[name]
        vision = Path(source["output"])
        response, registration = vision / "responses.npz", vision / "retina-registration.npz"
        for path, expected in (
            (response, source["responses_sha256"]),
            (registration, first["source_hashes"][str(registration)]),
        ):
            if sha256(path) != expected:
                raise ValueError("Original sensory input changed")
        with load_array(registration, allow_pickle=False) as x:
            uv = x["image_uv"].copy()
            if uv.shape != (721, 2) or x["view_order"].tolist() != [
                "front",
                "left",
                "right",
                "rear",
            ]:
                raise ValueError("Original view registration required")
        bins = np.minimum(((uv - uv.min(axis=0)) / np.ptp(uv, axis=0) * 4).astype(int), 3)
        weights = np.zeros((721, 16))
        for i in range(16):
            mask = (bins[:, 0] == i % 4) & (bins[:, 1] == i // 4)
            if mask.any():
                weights[mask, i] = 1 / mask.sum()
        with load_array(response, allow_pickle=False) as x:
            if x["additional_cell_types"].tolist() != ["L1", "L2", "Mi1", "Tm3", "Tm1", "Tm9"]:
                raise ValueError("Original retinal channel identities required")
            clocks = x["decision_monotonic_seconds"].copy()
            retinal = (np.asarray(x["additional_activity"]) @ weights).reshape(len(clocks), -1)
            motion = (np.asarray(x["motion_activity"]) @ weights).reshape(len(clocks), -1)
        features = np.concatenate([retinal, motion], axis=1)
        if features.shape != (739, 896) or not np.isfinite(features).all():
            raise ValueError("Exact original 14-channel four-view features required")
        predictions = {}
        for mode, folder, proof in (
            ("unweighted", original, first),
            ("robust", robust, second),
        ):
            artifact = folder / (name + "-both.npz")
            report = next(
                r for r in proof["reports"] if r["held_out"] == name and r["condition"] == "both"
            )
            if sha256(artifact) != report["artifact_sha256"]:
                raise ValueError("Original frozen fold checkpoint changed")
            with load_array(artifact, allow_pickle=False) as x:
                mean, scale, weight, centre, prior = (
                    x[k].copy() for k in ("mean", "scale", "weight", "centre", "prediction")
                )
            normalized = np.clip((features - mean) / scale, -5, 5)
            cases = [normalized]
            for view in range(4):
                modified = normalized.copy()
                modified[:, view * 96 : (view + 1) * 96] = 0
                modified[:, 384 + view * 128 : 384 + (view + 1) * 128] = 0
                cases.append(modified)
            cases.append(np.zeros_like(normalized))
            predicted = np.stack([centre + x @ weight for x in cases], axis=1)
            np.testing.assert_allclose(predicted[:, 0], prior, atol=1e-12, rtol=0)
            predictions[mode] = predicted
        np.savez_compressed(run / (name + ".npz"), time=clocks, **predictions)
        # Teacher target is accessed after every intervention prediction is fixed.
        if sha256(entry["targets_file"]) != entry["targets_sha256"]:
            raise ValueError("Original teacher diagnostic changed")
        with load_array(entry["targets_file"], allow_pickle=False) as x:
            np.testing.assert_array_equal(clocks, x["observation_time"])
            target, valid = x["preview"][:, 2, 1].copy(), x["preview_valid"][:, 2].copy()
        report = {"id": name, "valid": int(valid.sum()), "modes": {}}
        for mode, predicted in predictions.items():
            changes = np.mean(np.abs(predicted[valid, 1:] - predicted[valid, :1]), axis=0)
            rmse = np.sqrt(np.mean((predicted[valid] - target[valid, None]) ** 2, axis=0))
            report["modes"][mode] = {
                "mean_abs_change_per_m": changes.tolist(),
                "rmse_per_m": rmse.tolist(),
                "prediction_sha256": sha256(run / (name + ".npz")),
            }
        all_reports.append(report)
        print(json.dumps(report), flush=True)
    write_json(
        run / "result.json",
        {
            "status": "completed_image_grid_view_interventions",
            "conditions": [
                "original",
                "front_mean",
                "left_mean",
                "right_mean",
                "rear_mean",
                "all_mean",
            ],
            "reports": all_reports,
            "labels_only_after_predictions": True,
            "teacher_chord_not_instantaneous_steering": True,
            "same_track_not_independent": True,
            "physical_sync_verified": False,
            "actor_deployed": False,
            "goal_achieved": False,
        },
    )


if __name__ == "__main__":
    main()
