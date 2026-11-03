"""Paired-pose visual/body information probes; truth labels never enter actor tensors."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def ridge_probe(train, test, labels_train, labels_test, strength=.1):
    """Fixed ridge strength and training-only scaling; no test-set model selection."""
    mean = train.mean(0)
    scale = np.maximum(train.std(0), .05)
    x, z = (train - mean) / scale, (test - mean) / scale
    center = labels_train.mean(0)
    dual = np.linalg.solve(x @ x.T + strength * np.eye(len(x)), labels_train - center)
    predicted = z @ x.T @ dual + center
    return {"rmse": np.sqrt(np.mean((predicted - labels_test)**2, axis=0)).tolist(),
            "mean_label_baseline_rmse": np.sqrt(np.mean(
                (center - labels_test)**2, axis=0)).tolist(),
            "prediction": predicted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Optional compatible policy for frame comparison")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.evidence import begin_run, write_json
    from flybrain_drive.graph import sha256
    from flybrain_drive.imitation import teacher_action
    from flybrain_drive.learning import restore_policy
    from flybrain_drive.sensory_imitation import checked_environment
    from flybrain_drive.sensory_runtime import make_sensory_runtime

    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    policy, sealed = None, config
    if args.checkpoint:
        policy, sealed = restore_policy(args.checkpoint)
        policy.eval()
    train_x = [3, 18, 33, 48, 63, 78]
    test_x = [10, 25, 40, 55, 70, 85]
    offsets, headings = [-1., -.5, 0., .5, 1.], [-10., -5., 0., 5., 10.]
    run = begin_run(args.output, {"configuration": config,
                                 "checkpoint_sha256": (sha256(args.checkpoint)
                                                       if args.checkpoint else None),
                                 "diagnostic_script_sha256": sha256(__file__),
                                 "train_x": train_x, "test_x": test_x, "offsets": offsets,
                                 "heading_degrees": headings, "ridge_strength": .1,
                                 "scope": "static pose diagnostic, not driving or final test"})
    env = FlyBrainEnv(**checked_environment(config))
    rows, labels, cases, prediction = [], [], [], []
    try:
        bridge, packer, effective = make_sensory_runtime(env, config)
        if policy is not None and (
            effective["sensory_encoder"] != sealed["sensory_encoder"]
            or effective["sensory_preprocessing"] != sealed["sensory_preprocessing"]
        ):
            raise ValueError("Checkpoint sensory calibration differs from requested configuration")
        write_json(run / "eye-maps.json", {eye.side: eye.manifest() for eye in bridge.eyes})
        write_json(run / "sensory-calibration.json", effective["sensory_encoder"])
        for index, (split, x) in enumerate([("train", x) for x in train_x]
                                          + [("test", x) for x in test_x]):
            for lateral in offsets:
                for heading in headings:
                    obs, info = env.reset(seed=600101 + index, options={
                        "start_x": x, "lateral_offset": lateral, "heading_offset_degrees": heading})
                    episode = f"pose-{len(rows)}"
                    actor = bridge.observe(env, obs, episode_id=episode, sequence_id=0,
                                           received_monotonic_seconds=time.monotonic())
                    packed = packer.pack(actor)
                    rotation = env.data.xmat[env.model.body("chassis").id].reshape(3, 3)
                    frame = np.asarray(info["truth"]["road_frame"])
                    direction = float(np.arctan2(rotation[:, 0] @ frame[:, 1],
                                                  rotation[:, 0] @ frame[:, 0]))
                    label = [info["truth"]["lateral_error"], direction,
                             float(teacher_action(env, info, config["fixed_speed_mps"])[1])]
                    if policy is not None:
                        tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in packed.items()}
                        predictions = []
                        with torch.no_grad():
                            for mode in ("full", "no_vision", "no_body"):
                                masked = {k: v.clone() for k, v in tensors.items()}
                                if mode == "no_vision":
                                    masked["ray_values"].zero_()
                                    masked["ray_valid"].zero_()
                                if mode == "no_body":
                                    for key in (
                                        "organs", "organ_valid", "auxiliary", "auxiliary_valid",
                                        "applied_control", "applied_valid",
                                    ):
                                        masked[key].zero_()
                                mean, _, _ = policy(masked, policy.initial_state())
                                predictions.append(float(torch.tanh(mean)[0, 1]))
                        prediction.append(predictions)
                    rows.append(packed)
                    labels.append(label)
                    cases.append({"split": split, "start_x": x, "offset": lateral,
                                  "heading_degrees": heading})
            write_json(run / "progress.json", {"positions": index + 1, "samples": len(rows)})
            print(f"probe_positions={index+1} samples={len(rows)}", flush=True)
    finally:
        env.close()
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    labels = np.array(labels)
    predictions = np.array(prediction) if policy is not None else None
    train = np.array([c["split"] == "train" for c in cases])
    visual = (arrays["ray_values"] * arrays["ray_valid"][..., None]).reshape(len(rows), -1)
    body = np.concatenate([arrays[k].reshape(len(rows), -1) for k in
                           ("organs", "organ_valid", "auxiliary", "auxiliary_valid",
                            "applied_control", "applied_valid")], axis=1)
    probes = {}
    for name, features in (("visual", visual), ("body", body),
                           ("combined", np.concatenate([visual, body], axis=1))):
        result = ridge_probe(features[train], features[~train], labels[train], labels[~train])
        np.save(run / f"{name}-test-predictions.npy", result.pop("prediction"))
        probes[name] = result
    comparison = {"checkpoint_predictions": predictions} if predictions is not None else {}
    np.savez_compressed(run / "poses.npz",
                        **{"obs_" + k: v for k, v in arrays.items()}, **comparison,
                        teacher_only_labels=labels,
                        train_mask=train)
    write_json(run / "cases.json", cases)
    write_json(run / "summary.json", {
        "label_columns": ["lateral_m", "heading_rad", "teacher_steer_intent"],
        "prediction_modes": (["full", "no_vision", "no_body"] if predictions is not None else []),
        "ridge_probes": probes,
        "checkpoint_test_steer_rmse": (np.sqrt(np.mean(
            (predictions[~train] - labels[~train, 2, None])**2, axis=0)).tolist()
            if predictions is not None else None),
        "scope": "static single-frame held-out-position probe; no autonomous driving proof",
    })
    write_json(run / "result.json", {"status": "completed", "samples": len(rows),
                                     "poses_sha256": sha256(run / "poses.npz")})


if __name__ == "__main__":
    main()
