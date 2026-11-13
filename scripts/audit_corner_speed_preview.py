"""Join frozen camera predictions and audit preview-induced speed changes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from flybrain_drive.local_racing_line import plan_local_line, speed_bound_with_corner_preview
from train_vision_teacher_line import check, digest, jsonl, write


LIMITS = dict(clearance_m=1.1, max_curvature=.15, lateral_accel_mps2=4.,
              braking_mps2=4., speed_ceiling_mps=25.)
CORNER_SPEED_MPS = 14.
MODELS = ("linear", "flyvis", "linear_plus_flyvis")
DIRECTIONS = ("none", "left", "right")
SPLITS = {"heldout", "test_other", "external"}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def key(row):
    return row["run"], row["frame"]


def audit(boundaries, identity, forecast, out, revision):
    fingerprints = {}
    forecasts = {}
    for model in MODELS:
        path = forecast / f"{model}-predictions.jsonl"
        values = [r for r in rows(path) if r["split"] in SPLITS]
        forecasts[model] = {key(r): r for r in values}
        check(len(forecasts[model]) == len(values), "duplicate forecast capture")
        fingerprints[str(path)] = digest(path)
    reference = forecasts[MODELS[0]]
    for model in MODELS[1:]:
        check(forecasts[model].keys() == reference.keys(), "unpaired forecast set")
        for k, r in reference.items():
            check(all(forecasts[model][k][f] == r[f] for f in ("time", "split", "direction", "distance")),
                  "forecast truth mismatch")
    results = defaultdict(Counter)
    matched, outputs = set(), []
    for path in sorted(boundaries.glob("*-predictions.jsonl")):
        identity_path = identity / path.name
        if not identity_path.exists():
            continue
        for p in (path, identity_path):
            fingerprints[str(p)] = digest(p)
        predictions, captures = rows(path), rows(identity_path)
        check(len(predictions) == len(captures), f"unpaired boundary frames {path.name}")
        for boundary, capture in zip(predictions, captures):
            for field in ("episode_id", "observation_time_s", "source_speed_mps"):
                check(boundary[field] == capture[field], f"boundary clock mismatch {path.name} {field}")
            k = (capture["episode_id"], capture["capture_id"])
            if k not in reference:
                continue
            truth = reference[k]
            check(abs(truth["time"]-capture["observation_host_s"]) < 1e-6, "forecast clock mismatch")
            check(k not in matched, "duplicate boundary capture")
            matched.add(k)
            speed = boundary["source_speed_mps"]
            line, status = plan_local_line(boundary["road_geometry"]["sections"], speed, **LIMITS)
            for model in MODELS:
                r = forecasts[model][k]
                counters = results[(model, r["split"])]
                counters["matched"] += 1
                counters[status] += 1
                if line is None:
                    continue
                old = line["speed_bound_mps"][0]
                distance = r["predicted_distance_m"]
                check((distance is None) == (r["predicted"] == "none"), "forecast abstention contract")
                updated = speed_bound_with_corner_preview(
                    line, distance, corner_speed_mps=CORNER_SPEED_MPS, braking_mps2=LIMITS["braking_mps2"])
                new = float(updated[0])
                category = ("true_none" if r["direction"] == 0 else
                            "correct_direction" if r["predicted"] == DIRECTIONS[r["direction"]] else "wrong_direction")
                if new < old - 1e-9:
                    counters[f"lower_entry_{category}"] += 1
                    if new < speed <= old:
                        counters[f"new_brake_advisory_{category}"] += 1
                if np.any(updated < np.asarray(line["speed_bound_mps"])-1e-9):
                    counters[f"lower_anywhere_{category}"] += 1
                outputs.append(dict(run=k[0], frame=k[1], split=r["split"], model=model,
                    direction=r["predicted"], predicted_distance_m=distance, truth_direction=DIRECTIONS[r["direction"]],
                    entry_mps=speed, original_entry_bound_mps=old, preview_entry_bound_mps=new))
    check(matched, "no paired boundary and preview frames")
    jsonl(out/"advisories.jsonl", outputs)
    write(out/"result.json", dict(revision=revision, program_sha256=digest(Path(__file__)),
        planner_sha256=digest(Path(__file__).parents[1]/"src/flybrain_drive/local_racing_line.py"),
        input_sha256=fingerprints, seed=None, limits=LIMITS, corner_speed_mps=CORNER_SPEED_MPS,
        matched_captures=len(matched), unmatched_forecast_captures=len(reference.keys()-matched),
        results={model: {split: dict(counter) for (m, split), counter in results.items() if m == model}
                 for model in MODELS}, termination="saved-frame advisory audit completed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("boundaries", "identity", "forecast", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        audit(args.boundaries, args.identity, args.forecast, args.out, args.revision)
    except Exception as error:
        write(args.out/"failure.json", dict(termination=type(error).__name__, error=str(error)))
        raise
