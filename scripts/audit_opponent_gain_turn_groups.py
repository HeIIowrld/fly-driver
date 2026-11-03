"""Posthoc turn-stratified audit of the fixed TRAIN-only gain selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_learning import turn_groups
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    selection = Path(a.selection)
    proof = read(selection / "result.json")
    config = read(Path(a.training) / "manifest.json")["config"]["plan"]
    root = Path(config["data"])
    dataset = root / "dataset.json"
    predictions = selection / "predictions.npz"
    if (
        proof["status"] != "completed_opponent_gain_train_selection"
        or sha256(predictions) != proof["predictions_sha256"]
        or sha256(dataset) != proof["dataset_sha256"]
        or proof["selected_factor"] not in proof["factors"]
    ):
        raise ValueError("Exact fixed TRAIN selection and original dataset required")
    selected = proof["factors"].index(proof["selected_factor"])
    accum = {
        name: {"squared": np.zeros(2), "matching": np.zeros(2), "count": 0}
        for name in ("positive", "negative", "near_neutral")
    }
    with load_array(predictions, allow_pickle=False) as predictions_data:
        for entry in read(dataset)["episodes"]:
            if entry["role"] != "train":
                continue
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original TRAIN history changed")
            stem = entry["file"].removesuffix(".npz")
            with load_array(path, allow_pickle=False) as raw:
                np.testing.assert_array_equal(
                    predictions_data[stem + "_time"], raw["observation_time"]
                )
                truth, valid = raw["target"][:, 0].copy(), raw["target_valid"][:, 0].copy()
            out = predictions_data[stem + "_output"][:, [0, selected], 0]
            groups = turn_groups(truth, valid, threshold=0.01)
            for index, name in enumerate(accum):
                mask = groups == index
                accum[name]["count"] += int(mask.sum())
                accum[name]["squared"] += np.sum((out[mask] - truth[mask, None]) ** 2, axis=0)
                accum[name]["matching"] += np.sum(
                    np.sign(out[mask]) == np.sign(truth[mask, None]), axis=0
                )
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "selection_result_sha256": sha256(selection / "result.json"),
            "predictions_sha256": proof["predictions_sha256"],
            "dataset_sha256": sha256(dataset),
            "threshold": 0.01,
            "posthoc_diagnostic_only": True,
            "development_opened": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    groups = {}
    for name, value in accum.items():
        n = value["count"]
        if not n:
            raise ValueError("All three original target groups required")
        groups[name] = {
            "rows": n,
            "rmse_base_selected": np.sqrt(value["squared"] / n).tolist(),
            "sign_matches_base_selected": value["matching"].astype(int).tolist(),
        }
    write_json(
        run / "result.json",
        {
            "status": "completed_opponent_gain_turn_groups",
            "groups": groups,
            "selected_factor": proof["selected_factor"],
            "teacher_not_actor_input": True,
            "posthoc_diagnostic_only": True,
            "actor_deployed": False,
            "goal_achieved": False,
        },
    )
    print(json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
