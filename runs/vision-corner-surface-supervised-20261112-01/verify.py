"""Verify a frozen checkpoint, Silverstone replay parity, and paired warning outcomes."""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import json

import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
ROOT = RUNS.parent
SILVER = "ks_silverstone1967"


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def state(offer, row):
    if offer is None:
        return "no_warning"
    if int(row["label"]) != offer[0]:
        return "wrong_direction"
    low, high = row["truth_endpoints_m"]
    return "joint_hit" if low-5. <= offer[1] <= high+5. else "distance_error"


def main():
    training_path, result_path = OUT / "training.json", OUT / "result.json"
    training, result = read(training_path), read(result_path)
    assert training["excluded_circuits"] == [SILVER]
    assert SILVER not in result["training_circuits"] == training["training_circuits"]
    assert training["best_epoch"] == min(training["history"], key=lambda x:x["selection_score"])["epoch"]
    assert digest(OUT / "boundary.pt") == training["checkpoint_sha256"]
    sources = dict(training["source_sha256"])
    sources.update(result["source_sha256"])
    for filename, expected in sources.items():
        assert digest(ROOT / filename) == expected, filename
    paths = {
        "predictions_sha256": OUT / "image-only-predictions.npz",
        "rows_sha256": OUT / "silverstone-frames.jsonl",
        "sections_sha256": OUT / "observed-cross-sections.npz",
    }
    for key, path in paths.items():
        assert digest(path) == result[key], key
    with np.load(OUT / f"{SILVER}-front.npz", allow_pickle=False) as cache:
        assert set(cache.files) == {"images", "available", "run", "frame"}
    rows = [json.loads(line) for line in paths["rows_sha256"].read_text(encoding="utf8").splitlines()]
    old_path = RUNS / "vision-corner-extended-horizon-20261112-01" / f"{SILVER}-frames.jsonl"
    older = [json.loads(line) for line in old_path.read_text(encoding="utf8").splitlines()]
    keys = [(r["run"], r["frame"]) for r in rows]
    assert len(keys) == len(set(keys)) == result["frames"] == 1200
    for row, original in zip(rows, older, strict=True):
        for key in ("run", "frame", "truth", "label", "truth_endpoints_m"):
            assert row[key] == original[key]
        for mode in ("edge_far_wide", "t4t5_far_wide"):
            assert row["decisions"][mode] == original["final"][mode]
    for key in ("predictions_sha256", "sections_sha256"):
        with np.load(paths[key], allow_pickle=False) as data:
            assert list(zip(data["run"], data["frame"], strict=True)) == keys
            if key == "predictions_sha256":
                assert data["values"].shape == (1200, 13, 2, 4)
    summaries, frames = {}, {}
    right = [row for row in rows if row["truth"] == "approach" and row["label"] == "2"]
    assert len(right) == 13
    for mode in ("direct_fill", "direct_replace", "learned_row", "plane_row"):
        count = Counter()
        detail = []
        for row in right:
            comparison = {name: [state(row["decisions"][f"{name}_{m}"], row)
                                  for m in ("far_wide", mode)] for name in ("edge", "t4t5")}
            for name, (before, after) in comparison.items():
                count[f"{name}: {before} to {after}"] += 1
            if any(before != after for before,after in comparison.values()):
                detail.append(dict(frame=row["frame"], truth_m=row["truth_endpoints_m"], changes=comparison,
                                   offers=row["offers"], decisions=row["decisions"]))
        summaries[mode] = dict(count)
        frames[mode] = detail
    report = dict(status="verified", seed=training["seed"], source_revision=result["source_revision"],
                  source_files=len(sources), training_sha256=digest(training_path), result_sha256=digest(result_path),
                  verifier_sha256=digest(Path(__file__)), right_changes=summaries, changed_frames=frames,
                  termination="weights_inputs_and_1200_fixed_control_frames_verified")
    (OUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print("Frozen checkpoint, training exclusion and 1,200 Silverstone controls verified.")


if __name__ == "__main__":
    main()
