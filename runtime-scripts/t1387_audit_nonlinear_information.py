"""Independent float64 NumPy replay of fixed diagnostic controls; no fitting."""

import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

source = Path(
    "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1386-nonlinear-information-diagnosis"
)
root = Path(
    "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1360-v2-flight-episode-cache"
)
run = begin_run(
    "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1387-nonlinear-independent-audit",
    dict(
        scope=__doc__,
        script_sha256=sha256(__file__),
        source_result_sha256=sha256(source / "result.json"),
    ),
)
entries = json.loads((root / "dataset.json").read_text(encoding="utf8"))["episodes"]
arrays = []
for e in entries:
    assert e["role"] in ("train", "development")
    assert sha256(root / e["file"]) == e["sha256"]
    with load_array(root / e["file"], allow_pickle=False) as a:
        arrays.append(
            {
                k: a[k].copy()
                for k in ("scene_features", "target", "target_valid", "observation_time")
            }
        )
train = np.concatenate(
    [
        a["scene_features"][a["target_valid"][:, 0]]
        for e, a in zip(entries, arrays, strict=True)
        if e["role"] == "train"
    ]
).astype(np.float64)
with np.load(source / "normalization.npz") as norm:
    mean, scale = norm["mean"], norm["scale"]
assert np.array_equal(mean, train.mean(0))
assert np.array_equal(scale, np.maximum(train.std(0), 0.005))
result = json.loads((source / "result.json").read_text(encoding="utf8"))
saved = np.load(source / "predictions.npz")
maximum = 0.0
train_rmse = {}
for variant in ("linear_tanh", "hidden64_tanh"):
    state = {
        k: v.numpy().astype(np.float64)
        for k, v in torch.load(
            source / f"{variant}.pt", map_location="cpu", weights_only=True
        ).items()
    }
    errors = []
    for entry, a, report in zip(entries, arrays, result["metrics"][variant], strict=True):
        assert report["file"] == entry["file"]
        normalized = (
            np.clip((a["scene_features"] - mean) / scale, -5, 5)
            .astype(np.float32)
            .astype(np.float64)
        )
        for condition in ("intact", "motion_train_mean", "intensity_train_mean", "all_train_mean"):
            inputs = normalized.copy()
            channels = inputs.reshape(len(inputs), 4, 9, 16)
            if condition == "motion_train_mean":
                channels[:, :, 1:] = 0
            elif condition == "intensity_train_mean":
                channels[:, :, :1] = 0
            elif condition == "all_train_mean":
                inputs[:] = 0
            prediction = np.tanh(inputs @ state["0.weight"].T + state["0.bias"])
            if variant == "hidden64_tanh":
                prediction = np.tanh(prediction @ state["2.weight"].T + state["2.bias"])
            prediction = prediction[:, 0]
            stored = saved[f"{variant}_{condition}_{entry['file'].removesuffix('.npz')}"]
            maximum = max(maximum, float(np.max(np.abs(prediction - stored))))
            valid = a["target_valid"][:, 0]
            rmse = float(np.sqrt(np.mean((prediction[valid] - a["target"][valid, 0]) ** 2)))
            assert abs(rmse - report["conditions"][condition]["full"]["rmse"]) < 3e-6
            if condition == "intact" and entry["role"] == "train":
                errors.append(prediction[valid] - a["target"][valid, 0])
    train_rmse[variant] = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
assert maximum < 3e-6
write_json(
    run / "result.json",
    dict(
        status="passed_independent_float64_replay",
        maximum_prediction_error=maximum,
        tolerance=3e-6,
        train_rmse=train_rmse,
        original_train_normalization_exact=True,
        original_episode_hashes_verified=len(entries),
        evaluation_role_arrays_opened=False,
        optimizer_updates=0,
        ui_actions_sent=0,
        actuator_messages_sent=0,
        goal_achieved=False,
    ),
)
print(json.dumps(dict(maximum_prediction_error=maximum, train_rmse=train_rmse)))
