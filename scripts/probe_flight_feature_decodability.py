"""Train-only ridge probes of allowed feature information; not a replacement driving policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_learning import turn_groups
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    receipt, config = read(root / "result.json"), read(root / "manifest.json")["config"]
    data_root = Path(config["plan"]["data"])
    if (sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
            or sha256(data_root / "dataset.json") != config["dataset_manifest_sha256"]):
        raise ValueError("Original fixed model normalization and training data required")
    alphas = [.01, .1, 1.]
    run = begin_run(args.output, dict(arguments=vars(args), seed=863161,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        dataset_manifest_sha256=config["dataset_manifest_sha256"],
        ridge_alphas=alphas, selection="none; report every predeclared feature set and alpha",
        feature_sets=["intensity", "motion", "visual", "body", "visual_body"],
        roles="train only; development and fresh evaluation arrays never opened",
        labels="original left-turn intent; equal class totals at existing threshold .01",
        scope="static linear information/fit probe, not flight mechanism or driving success",
        actor_qualified=False, actuator_messages_sent=0))
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    mean = payload["model"]["scene_command.mean"].numpy().astype(np.float64)
    scale = payload["model"]["scene_command.scale"].numpy().astype(np.float64)
    body_scale = payload["model"]["body_scale"].numpy().astype(np.float64)
    matrices = {key: [] for key in ["intensity", "motion", "visual", "body", "visual_body"]}
    targets, groups, sources = [], [], []
    for entry in read(data_root / "dataset.json")["episodes"]:
        if entry["role"] != "train":
            continue
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original training episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            target, valid = archive["target"][:, 0], archive["target_valid"][:, 0]
            visual = np.clip((archive["scene_features"].astype(np.float64) - mean) / scale, -5, 5)
            structured = visual.reshape(-1, 4, 9, 16)
            body_valid = archive["body_valid"]
            body = np.column_stack([np.tanh(archive["body"].astype(np.float64) / body_scale)
                                    * body_valid, body_valid])
            goal = archive["goals"][:, 1:2]
            values = dict(intensity=structured[:, :, :1].reshape(len(visual), -1),
                          motion=structured[:, :, 1:].reshape(len(visual), -1),
                          visual=visual, body=body, visual_body=np.column_stack([visual, body]))
            for name, value in values.items():
                matrices[name].append(np.column_stack([value[valid], goal[valid]]))
            targets.append(target[valid].astype(np.float64))
            groups.append(turn_groups(target, valid)[valid])
            sources.append(dict(file=entry["file"], observations=len(target),
                                labels=int(valid.sum()),
                                source_sha256=entry["sha256"]))
    y, strata = np.concatenate(targets), np.concatenate(groups)
    counts = [int(np.count_nonzero(strata == group)) for group in range(3)]
    if not all(counts):
        raise ValueError("All three observed training strata required")
    weights = np.array([1 / (3 * counts[g]) for g in strata])
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12)
    y_mean = float(weights @ y)
    reports = []
    for name, blocks in matrices.items():
        x = np.concatenate(blocks)
        x_mean = weights @ x
        centered, centered_y = x - x_mean, y - y_mean
        gram = centered.T @ (weights[:, None] * centered)
        rhs = centered.T @ (weights * centered_y)
        eigenvalue, eigenvector = np.linalg.eigh(gram)
        for alpha in alphas:
            coefficient = eigenvector @ ((eigenvector.T @ rhs) / (eigenvalue.clip(min=0) + alpha))
            intercept = float(y_mean - x_mean @ coefficient)
            prediction = x @ coefficient + intercept
            if not np.isfinite(prediction).all():
                raise ValueError("Finite train-only linear probe required")
            metrics = {}
            for group, label in enumerate(["left", "right", "near_neutral"]):
                chosen = strata == group
                metrics[label] = dict(count=counts[group],
                    rmse=float(np.mean((prediction[chosen] - y[chosen]) ** 2) ** .5),
                    zero_turn_rmse=float(np.mean(y[chosen] ** 2) ** .5),
                    prediction_mean=float(prediction[chosen].mean()),
                    target_mean=float(y[chosen].mean()),
                    sign_correct_fraction=float((np.sign(prediction[chosen])
                                                 == np.sign(y[chosen])).mean()))
            file = f"{name}-alpha{alpha}.npz"
            np.savez_compressed(run / file, coefficient=coefficient, intercept=intercept,
                                prediction=prediction, target=y, strata=strata)
            reports.append(dict(features=name, alpha=alpha, parameters=len(coefficient) + 1,
                metrics=metrics, prediction_unclipped=True, file=file, sha256=sha256(run / file)))
        write_json(run / "progress.json", dict(feature_sets_completed=name))
    write_json(run / "result.json", dict(status="completed_train_only_information_probe",
        sources=sources, counts=counts, reports=reports, models_selected=0,
        original_neural_feature_histories_preserved=True, source_arrays_modified=False,
        temporal_encoder_refitted=False, internal_flight_graph_used_by_probe=False,
        development_arrays_opened=False, fresh_evaluation_arrays_opened=False,
        generalization_verified=False, physiological_recognition_verified=False,
        actor_qualified=False, actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(status="completed", probes=len(reports), counts=counts)))


if __name__ == "__main__":
    main()
