"""Compare readout histories on a fixed, previously extracted next-corner corpus."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from train_next_corner import HORIZON, check, digest, metrics, predict, read, select_model, write


def run(source, out):
    row_file = source / "rows.jsonl"
    feature_file = source / "flyvis-features.npz"
    receipt = read(source / "flyvis-source.json")
    check(digest(row_file) == receipt["rows_sha256"], "row receipt mismatch")
    check(digest(feature_file) == receipt["features_sha256"], "feature receipt mismatch")
    rows = [json.loads(line) for line in row_file.read_text(encoding="utf-8").splitlines()]
    excluded = {(r["run"], i) for r in rows if r["split"] == "excluded"
                for i in range(r["frame"] - 2, r["frame"] + 1)}
    with np.load(feature_file, allow_pickle=False) as archive:
        available = archive["available"]
        features = archive["features"]
    check(available.shape == (len(rows),) and available.dtype == bool, "availability contract")
    check(features.shape == (len(rows), 3 * 4 * 8 * 16), "FlyVis history layout")
    keep = available & np.array([
        r["split"] not in ("train", "validation") or not any(
            (r["run"], i) in excluded for i in range(r["frame"] - 2, r["frame"] + 1))
        for r in rows
    ])
    rows = [r for r, valid in zip(rows, keep) if valid]
    history = features[keep].astype(np.float64).reshape(len(rows), 3, -1)
    check(np.isfinite(history).all(), "nonfinite FlyVis history")
    results = {}
    variants = {
        "full": history.reshape(len(rows), -1),
        "latest": history[:, -1],
        "mean": history.mean(axis=1),
        "changes": np.diff(history, axis=1).reshape(len(rows), -1),
    }
    for name, x in variants.items():
        classifier, regressor, bias, selection = select_model(x, rows)
        prediction = (predict(classifier, x) + np.array([bias, 0., 0.])).argmax(axis=1)
        distance = predict(regressor, x).clip(0, HORIZON)
        scores = {}
        for split in ("validation", "heldout", "test_other", "external"):
            mask = np.array([r["split"] == split for r in rows])
            scores[split] = metrics([r for r, valid in zip(rows, mask) if valid],
                                    prediction[mask], distance[mask])
        results[name] = dict(selection=selection, results=scores)
    reference = read(source / "result.json")["models"]["flyvis"]["results"]
    for split, score in results["full"]["results"].items():
        check(score["confusion"] == reference[split]["confusion"]
              and np.isclose(score["distance_mae_m"], reference[split]["distance_mae_m"]),
              f"full history failed to reproduce {split}")
    revision = subprocess.check_output(["git", "-c", f"safe.directory={Path.cwd().as_posix()}",
                                       "rev-parse", "HEAD"], text=True).strip()
    write(out / "result.json", dict(
        source=str(source), rows_sha256=receipt["rows_sha256"], features_sha256=receipt["features_sha256"],
        reference_sha256=digest(source / "result.json"), revision=revision,
        program_sha256=digest(Path(__file__)),
        training_code_sha256=digest(Path(__file__).with_name("train_next_corner.py")),
        randomness="none", termination="paired readout evaluation completed", models=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.source, args.out)
    except Exception as error:
        write(args.out / "failure.json", dict(termination=type(error).__name__, error=str(error)))
        raise
