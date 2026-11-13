"""Compare causal distance correction on complete, verified corner timelines."""
from collections import deque
from pathlib import Path

import numpy as np

from evaluate_fused_corner import frame_key, read_rows
import hinge
from profile_replay import CALIBRATION_TRACKS, TRACKS
from score_corner_captures import adjacent, summarize
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import compare_tracks, score

OUT = Path(__file__).parent
MOTION = Path("runs/vision-corner-edge-agreement-20261111-01")
SHAPE = Path("runs/vision-corner-edge-shape-readout-20261111-01")
LEARNED = Path("runs/vision-corner-learned-edge-20261111-01")
SETTINGS = {
    "prior25": ("blend", .25, 8.),
    "prior50": ("blend", .50, 8.),
    "prior75": ("blend", .75, 8.),
    "median3": ("median", 3, 8.),
    "bounded2": ("bounded", 2., 8.),
    "bounded4": ("bounded", 4., 8.),
    "long_prior50": ("long_blend", .50, 8.),
    "long_bounded2": ("long_bounded", 2., 8.),
}
DIRECTIONS = {"left": 1, "right": 2}


def raw_observation(row):
    direction = DIRECTIONS.get(row["state"])
    if direction is None:
        return None
    distance = row["predicted_distance_m"]
    if distance is None or not 0. <= distance <= 20.:
        return None
    return direction, float(distance)


def filter_distances(rows, travel, setting):
    kind, weight, reset = setting
    previous_row = None
    previous_side = previous_distance = previous_measurement = None
    history = deque(maxlen=int(weight) - 1 if kind == "median" else 1)
    position = 0.
    for row, step in zip(rows, travel, strict=True):
        observation = raw_observation(row)
        if observation is None:
            previous_side, previous_distance = None, None
            history.clear()
            yield row
            continue
        side, measured = observation
        prediction = None
        if (side == previous_side and adjacent(previous_row, row) and 0. <= step <= 12.):
            predicted = previous_distance - step
            # A passed start or incompatible innovation begins a new visual hypothesis.
            use_prior = not kind.startswith("long_") or min(previous_measurement, measured) >= 10.
            if use_prior and predicted >= 0. and abs(predicted - measured) <= reset:
                prediction = predicted
        if prediction is None:
            history.clear()
            position = 0.
            estimate = measured
        else:
            position += step
            if kind in ("blend", "long_blend"):
                estimate = weight * prediction + (1. - weight) * measured
            elif kind == "median":
                estimate = np.median([*history, measured + position]) - position
            else:
                estimate = prediction + np.clip(measured - prediction, -weight, weight)
        history.append(measured + position)
        estimate = float(np.clip(estimate, 0., 20.))
        yield score(row, (side, estimate)) if estimate != measured else row
        previous_row, previous_side, previous_distance, previous_measurement = row, side, estimate, measured


def verified_predictions(path, hashes):
    rows, receipt = path / "rows.jsonl", path / "result.json"
    if digest(rows) != read(receipt)["rows_sha256"]:
        raise ValueError(f"Prediction rows changed: {path}")
    hashes.update((rows, receipt))
    return read_rows(rows)


def supplementary(rows, track, shape, proposals):
    for row in rows:
        key = track, *frame_key(row)
        candidate = proposals.get(key)
        confidence = shape.get(key)
        if not hinge.recoverable(row) or candidate is None or confidence is None or not confidence["usable"]:
            yield row
            continue
        side = candidate[1]
        p = confidence["near_scores"]
        if p[side - 1] >= .85 and p[side - 1] > p[2 - side]:
            yield dict(score(row, (side, candidate[2])), abstain_reason=None)
        else:
            yield row


def evaluate(group, rows):
    result = compare_tracks(group["original"], rows, group["rules"])
    summaries = [summarize([r for r in rows if r["run"] == run])
                 for run in dict.fromkeys(r["run"] for r in rows)]
    alerts = [r for r in rows if r["direction_hit"]]
    result["absolute_mae_m"] = float(np.mean([r["absolute_distance_error_m"] for r in alerts])) if alerts else None
    result["stable_events"] = [dict(run=run, physical_corner=e["physical_corner"], first_frame=e["first_frame"],
                                    lead_m=e["first_stable_lead_m"])
                               for run, summary in zip(dict.fromkeys(r["run"] for r in rows), summaries, strict=True)
                               for e in summary["events"] if e["first_stable_frame"] is not None]
    return result


def choose(reviews, tracks):
    ranked = []
    baseline = reviews["visual"]
    for label, scores in reviews.items():
        changes = []
        for track in tracks:
            old, new = baseline[track], scores[track]
            a, b = (r["distances"]["candidate"] for r in (old, new))
            if (new["metrics"]["candidate"]["joint"] < old["metrics"]["candidate"]["joint"]
                    or len(new["stable_events"]) < len(old["stable_events"])
                    or new["shared_pairs"] != old["shared_pairs"] or b["receding"] > a["receding"]):
                break
            if a["mae_m"] is not None:
                if b["mae_m"] > a["mae_m"] + 1e-9:
                    break
                changes.append(a["mae_m"] - b["mae_m"])
        else:
            ranked.append(((sum(changes), label == "visual", label), label))
    return max(ranked)[1]


def read_travel(track, rows, hashes):
    path = MOTION / f"{track}-motion.npz"
    proof = path.with_suffix(".json")
    receipt = read(proof)
    if digest(path) != receipt["motion_sha256"]:
        raise ValueError(f"Relative-motion source changed: {track}")
    for filename, sha in receipt["input_sha256"].items():
        source = Path(filename)
        if digest(source) != sha:
            raise ValueError(f"Relative-motion provenance changed: {source}")
        hashes.add(source)
    hashes.update((path, proof))
    with np.load(path) as data:
        if list(zip(data["run"], data["frame"], strict=True)) != [frame_key(row) for row in rows]:
            raise ValueError(f"Motion frame order differs: {track}")
        return data["motion"]


def run():
    hashes = {Path(__file__), hinge.OUT / "hinge.py", Path("scripts/score_corner_captures.py"),
              Path("scripts/evaluate_independent_corner.py"),
              Path("runs/vision-corner-onset-support-20261111-01/transfer.py")}
    shape = {(r["track"], *frame_key(r)):r for r in verified_predictions(SHAPE, hashes) if r["mode"] == "aligned"}
    learned = {(r["track"], *frame_key(r)):r["proposal"] for r in verified_predictions(LEARNED, hashes)}
    combined_receipt = SHAPE / "joint-evidence-diagnostic.json"
    combined = read(combined_receipt)
    hashes.add(combined_receipt)
    reports = {base:{label:{} for label in ("visual", *SETTINGS)} for base in ("fixed", "joint")}
    changes = []
    for track in TRACKS:
        group = hinge.group(track, hashes)
        travel = read_travel(track, group["rows"], hashes)
        extended = list(supplementary(group["rows"], track, shape, learned))
        for base, baseline in (("fixed", group["rows"]), ("joint", extended)):
            base_report = evaluate(group, baseline)
            if base == "joint" and base_report["metrics"] != combined["tracks"][track]["comparison"]["metrics"]:
                raise ValueError(f"Joint evidence replays differently: {track}")
            reports[base]["visual"][track] = base_report
            for label, setting in SETTINGS.items():
                rows = list(filter_distances(baseline, travel, setting))
                report = evaluate(group, rows)
                counts = ("left", "right", "outside", "missing")
                if any(report["metrics"]["candidate"][n] != base_report["metrics"]["candidate"][n] for n in counts):
                    raise ValueError(f"Distance filter changed direction or presence: {track}/{label}")
                reports[base][label][track] = report
                for raw, updated in zip(baseline, rows, strict=True):
                    if raw is not updated:
                        changes.append(dict(track=track, base=base, method=label, run=raw["run"], frame=raw["frame"],
                                            truth=raw["truth"], label=raw["label"],
                                            truth_distance_m=raw["truth_distance_m"],
                                            original_m=raw["predicted_distance_m"], updated_m=updated["predicted_distance_m"]))
    choices = {}
    for track in TRACKS:
        training = tuple(name for name in CALIBRATION_TRACKS if name != track)
        choices[track] = dict(calibration_circuits=training, results={})
        for base, grid in reports.items():
            selected = choose(grid, training)
            choices[track]["results"][base] = dict(selected=selected, methods={n:r[track] for n, r in grid.items()})
            print(track, base, selected, grid[selected][track]["metrics"]["candidate"],
                  grid[selected][track]["distances"]["candidate"], flush=True)
    path = OUT / "changes.jsonl"
    jsonl(path, changes)
    write(OUT / "result.json", dict(tracks=choices, changes_sha256=digest(path),
          parameters=dict(distance_settings=SETTINGS, seed=None),
          source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="motion_anchored_distance_replays_complete"))


if __name__ == "__main__":
    run()
