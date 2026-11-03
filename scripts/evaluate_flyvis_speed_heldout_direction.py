"""Exploratory direction decoding of saved central FlyVis edge responses."""

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
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    result, stimulus = read(source / "result.json"), read(source / "stimulus.json")
    if result["status"] != "completed" or sha256(source / "responses.npz") != result["responses_sha256"]:
        raise ValueError("Original completed FlyVis responses with matching hash required")
    rows = stimulus["parameters"]
    angles = sorted({r["angle"] for r in rows})
    speeds = sorted({r["speed"] for r in rows})
    polarities = sorted({r["intensity"] for r in rows})
    if (stimulus["edge_protocol"] != "author-grid" or stimulus["dt"] != .005
            or angles != list(range(0, 360, 30)) or len(speeds) != 6
            or polarities != [0, 1] or len(rows) != 144
            or len({(r["angle"], r["speed"], r["intensity"]) for r in rows}) != 144):
        raise ValueError("Complete pinned 12 x 6 x 2 edge grid required")
    with load_array(source / "responses.npz", allow_pickle=False) as arrays:
        names = arrays["central_cell_types"].tolist()
        indices = [names.index(f"{kind}{direction}") for kind in ("T4", "T5") for direction in "abcd"]
        activity, lengths = arrays["activity"][:, :, indices].copy(), arrays["valid_frames"].copy()
    # Single predeclared feature per cell: maximum change from mean pre-stimulus activity.
    features = []
    for i, row in enumerate(rows):
        count = int(lengths[i])
        end = min(count, int(np.floor((1.0 + row["t_stim"]) / .005)) + 1)
        if count > activity.shape[1] or end <= 200 or not np.isfinite(activity[i, :count]).all():
            raise ValueError("Invalid original finite stimulus history")
        baseline = activity[i, :200].mean(axis=0, dtype=np.float64)
        features.append(activity[i, 200:end].max(axis=0) - baseline)
    features = np.asarray(features, dtype=np.float64)
    theta = np.deg2rad([r["angle"] for r in rows])
    targets = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    records = []
    for polarity in polarities:
        for speed in speeds:
            train = np.array([r["intensity"] == polarity and r["speed"] != speed for r in rows])
            test = np.array([r["intensity"] == polarity and r["speed"] == speed for r in rows])
            x = features[train]
            mean, scale = x.mean(axis=0), x.std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.)
            x = np.column_stack((np.ones(train.sum()), (x - mean) / scale))
            xt = np.column_stack((np.ones(test.sum()), (features[test] - mean) / scale))
            regularizer = np.eye(x.shape[1]); regularizer[0, 0] = 0
            weights = np.linalg.solve(x.T @ x + regularizer, x.T @ targets[train])
            predictions = xt @ weights
            pred_angle = np.mod(np.rad2deg(np.arctan2(predictions[:, 1], predictions[:, 0])), 360.)
            truth = np.rad2deg(theta[test])
            error = np.abs((pred_angle - truth + 180.) % 360. - 180.)
            for index, actual, predicted, diff in zip(np.flatnonzero(test), truth, pred_angle, error, strict=True):
                records.append(dict(index=int(index), polarity=polarity, speed=speed,
                                    angle=float(actual), predicted_angle=float(predicted),
                                    absolute_circular_error=float(diff)))
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), source_responses_sha256=result["responses_sha256"],
        stimulus_sha256=sha256(source / "stimulus.json"), seed=None, features="central T4a-d,T5a-d peak minus pre-stimulus mean",
        training="separate polarity; leave one stimulus speed out; train-only normalization; ridge alpha 1",
        evaluation="continuous circular angle error; +/-15 and +/-90 degrees", exploratory=True))
    write_json(run / "predictions.json", records)
    errors = np.array([r["absolute_circular_error"] for r in records])
    summaries = []
    for polarity in polarities:
        for speed in speeds:
            e = np.array([r["absolute_circular_error"] for r in records if r["polarity"] == polarity and r["speed"] == speed])
            summaries.append(dict(polarity=polarity, speed=speed, n=len(e), median_error=float(np.median(e)),
                                  within_15=int((e <= 15).sum()), within_90=int((e < 90).sum())))
    write_json(run / "result.json", dict(status="completed", predictions_sha256=sha256(run / "predictions.json"),
        n=len(errors), median_error=float(np.median(errors)), mean_error=float(np.mean(errors)),
        within_15=int((errors <= 15).sum()), within_90=int((errors < 90).sum()), folds=summaries,
        biological_validity_verified=False, flight_circuit_direction_readout_tested=False,
        actor_deployed=False, vehicle_control_sent=False))
    print(json.dumps(read(run / "result.json")))


if __name__ == "__main__":
    main()
