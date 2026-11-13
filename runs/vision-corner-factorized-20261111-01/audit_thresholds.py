"""Enumerate all observed operating points against development baselines."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
root = out.parent
proof = read(out/"result.json")
predictions = out/"predictions.jsonl"
assert digest(predictions) == proof["prediction_sha256"]


def index(path):
    return {(r["run"], r["frame"]): r for r in map(json.loads, path.read_text().splitlines())}


monza = index(root/"vision-monza-two-lap-presence-20261111-01"/"rows.jsonl")
magione = index(root/"vision-corner-independent-runner-rehearsal-20261111-01"/
                "replay"/"parent"/"score"/"rows.jsonl")
imola = index(root/"vision-corner-imola-full-parent-20261111-01"/"rows.jsonl")
sources = {
    "monza_validation": monza, "magione_transfer": magione, "imola_negative_transfer": imola,
    "magione_corner_2_and_unassigned": magione, "imola_holdout_recordings": imola,
}
values = defaultdict(list)
for row in map(json.loads, predictions.read_text().splitlines()):
    if row["distance_m"] > 20:
        continue
    cohort = row["cohort"]
    truth = sources[cohort][row["run"], row["frame"]]
    if cohort == "monza_validation":
        label, distance = truth["direction"], truth["distance"]
        category = "straight" if not label else ("approach" if 4 <= distance <= 20 else "out_of_range")
        distances = [distance] if distance is not None else []
    else:
        category = truth["truth"]
        label = int(truth["label"]) if truth["label"] in ("1", "2") else 0
        distances = truth["truth_endpoints_m"]
    side = 2 if row["right_minus_left"] > 0 else 1
    hit = category == "approach" and label == side
    joint = hit and max(abs(row["distance_m"] - d) for d in distances) <= 5
    values[row["model"]].append((row["presence_score"], cohort, np.array(
        [hit and label == 1, hit and label == 2, joint, category in ("straight", "out_of_range")], dtype=int)))

references = {c: np.array([r["left"], r["right"], r["joint"], r["straight"] + r["out_of_range"]])
              for c, r in proof["summary"]["previous_candidate"].items()}
result = {}
for model, rows in values.items():
    counts = {c: np.zeros(4, dtype=int) for c in sorted({r[1] for r in rows})}
    points = defaultdict(list)
    for score, cohort, increments in rows:
        points[score].append((cohort, increments))
    feasible = []
    for threshold, changed in sorted(points.items(), reverse=True):
        for cohort, increments in changed:
            counts[cohort] += increments
        if all((v[:3] >= references[c][:3]).all() and v[3] <= references[c][3] for c, v in counts.items()):
            feasible.append(dict(threshold=threshold, metrics={c: v.tolist() for c, v in counts.items()}))
    result[model] = dict(observed_thresholds=len(points), matching_or_better=len(feasible),
                         first_passing=feasible[0] if feasible else None)
write(out/"threshold-audit.json", dict(models=result, baseline="previous blended, ordinal distance",
     conditions="each cohort preserves left, right, and joint hits and does not increase straight plus outside alerts",
     metrics_order=["left", "right", "joint", "combined false alerts"],
     fit_result_sha256=digest(out/"result.json"), audit_source_sha256=digest(Path(__file__))))
for model, r in result.items():
    print(model, r["observed_thresholds"], r["matching_or_better"])
