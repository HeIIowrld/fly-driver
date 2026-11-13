"""Sensitivity of the frozen background score on development features only."""
from collections import Counter
from pathlib import Path

import numpy as np

from infer_frozen_next_corner import load_readout
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
root = out.parent
pins = root / "vision-frozen-next-corner-20261110-01/pins.json"
model, threshold = load_readout(read(pins))
inputs = [pins, Path(__file__)]
rows = []

def load(path):
    return list(map(__import__("json").loads, path.read_text(encoding="utf8").splitlines()))

def check(path, receipt, key):
    assert digest(path) == read(receipt)[key], path
    inputs.extend((path, receipt))

magione = root / "vision-magione-left-readout-stages-20261111-02/rows.jsonl"
check(magione, magione.with_name("result.json"), "rows_sha256")
for row in load(magione):
    rows.append(dict(cohort="magione_development", run=row["run"], frame=row["frame"],
                     actual=row["actual"], truth_m=row["onset_m"], near=row["near"],
                     scores=row["scores"], present=row["present"], distance_m=row["ordinal_m"],
                     available=True))

monza = root / "vision-monza-two-lap-presence-20261111-01"
labels = monza / "rows.jsonl"
features = monza / "features.npz"
proof = monza / "flyvis-result.json"
check(labels, proof, "rows_sha256")
check(features, proof, "features_sha256")
fold = load(labels)
with np.load(features, allow_pickle=False) as packed:
    direct, reflected, valid = (packed[k] for k in ("direct", "reflected", "reflected_available"))
for i, label in enumerate(fold):
    scores, distance, presence = model(direct[i:i+1], reflected[i:i+1] if valid[i] else None,
                                       return_scores=True)
    rows.append(dict(cohort="monza_" + label["split"], run=label["run"], frame=label["frame"],
                     actual=label["direction"], truth_m=label["distance"],
                     near=bool(label["direction"] and label["distance"] is not None
                               and 4 <= label["distance"] <= 20),
                     scores=scores[0].tolist(), present=bool(presence[0] >= threshold),
                     distance_m=float(distance[0]), available=True))

imola = root / "vision-imola-hard-negative-presence-20261110-01"
proof = imola / "result.json"
check(imola / "imola-features.npz", proof, "feature_sha256")
check(imola / "imola-rows.jsonl", proof, "imola_rows_sha256")
with np.load(imola / "imola-features.npz", allow_pickle=False) as packed:
    direct, reflected = packed["features"].transpose(1, 0, 2)
for i, label in enumerate(load(imola / "imola-rows.jsonl")):
    scores, distance, presence = model(direct[i:i+1], reflected[i:i+1], return_scores=True)
    rows.append(dict(cohort="imola_development_negative", run=label["run"], frame=label["frame"],
                     actual=None, truth_m=None, near=False, scores=scores[0].tolist(),
                     present=bool(presence[0] >= threshold), distance_m=float(distance[0]), available=True))

def prediction(row, margin):
    scores = np.array(row["scores"])
    scores[0] -= margin
    side = int(scores.argmax()) if np.count_nonzero(scores == scores.max()) == 1 else 0
    return side if row["present"] and row["distance_m"] <= 20 else 0

summary = {}
for cohort in sorted({r["cohort"] for r in rows}):
    group = [r for r in rows if r["cohort"] == cohort]
    summary[cohort] = {}
    for margin in (0., .01, .02, .04, .08):
        counts = Counter(frames=len(group))
        added = []
        for row in group:
            predicted = prediction(row, margin)
            if row["near"]:
                counts[f"near_{row['actual']}"] += 1
                counts[f"hit_{row['actual']}"] += predicted == row["actual"]
                counts["joint"] += bool(predicted == row["actual"]
                                        and abs(row["distance_m"] - row["truth_m"]) <= 5)
            else:
                counts["outside"] += bool(predicted)
            if predicted != prediction(row, 0.):
                added.append(dict(run=row["run"], frame=row["frame"], actual=row["actual"],
                                  truth_m=row["truth_m"], predicted=predicted))
        summary[cohort][str(margin)] = dict(counts=counts, changed=added)
jsonl(out / "rows.jsonl", rows)
write(out / "result.json", dict(summary=summary, rows_sha256=digest(out / "rows.jsonl"),
      hashes={str(p): digest(p) for p in inputs}, seed=None, revision=read(pins)["revision"],
      scope="All cohorts were previously consumed for development; no independent evaluation",
      candidate_promoted=False, termination="development sensitivity grid exported"))
for cohort, variants in summary.items():
    print(cohort, {margin: row["counts"] for margin, row in variants.items()})
