"""Check whether a learned visual readout corroborates geometric corner alerts."""
from collections import defaultdict, deque
import importlib.util
from pathlib import Path

from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
runs = out.parent
temporal = runs/"vision-corner-temporal-boundaries-20261111-01"
factorized = runs/"vision-corner-factorized-20261111-01"
uncertainty = runs/"vision-corner-boundary-uncertainty-20261111-01"
spec = importlib.util.spec_from_file_location("temporal_replay", temporal/"replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
factor_receipt = read(factorized/"result.json")
base_receipt = read(temporal/"result.json")
bounds_receipt = read(uncertainty/"result.json")
assert digest(temporal/"replay.py") == base_receipt["source_sha256"]
geometry = replay.read_rows(temporal/"geometry.jsonl", base_receipt["geometry_sha256"])
bounds = replay.read_rows(uncertainty/"bounds.jsonl", bounds_receipt["bounds_sha256"])
scores = replay.read_rows(factorized/"predictions.jsonl", factor_receipt["prediction_sha256"])
keys = [replay.key(row) for row in geometry]
assert keys == [replay.key(row) for row in bounds]
assert len(keys) == len(set(keys))

visual = defaultdict(dict)
for row in scores:
    if row["cohort"] not in ("magione_transfer", "imola_negative_transfer"):
        continue
    key = replay.key(row)
    assert key not in visual[row["model"]]
    visual[row["model"]][key] = row

geometry_choices = {}
for mode in ("unmodified", "whole_shape_min"):
    stream = geometry if mode == "unmodified" else [
        dict(r, curvature={"history_0.24": r["curvature"].get(mode, [])}) for r in bounds]
    for cut in (.002, .004):
        geometry_choices[f"{mode}_{cut}"] = replay.confirmed_predictions(stream, cut, 3, 1.)

paths = {
    "magione": runs/"vision-corner-independent-runner-rehearsal-20261111-01"/"replay"/"candidate"/"score"/"rows.jsonl",
    "imola": runs/"vision-corner-imola-full-blended-20261111-01"/"rows.jsonl",
}
reference = {track: replay.read_rows(path, base_receipt["scoring_inputs"][str(path)])
             for track, path in paths.items()}
baselines = {track: replay.reports(rows) for track, rows in reference.items()}
baseline = {track: replay.compact(report) for track, report in baselines.items()}

def increment(row):
    return (int(row["direction_hit"] and row["label"] == "1"),
            int(row["direction_hit"] and row["label"] == "2"), int(row["joint_hit"]),
            int(row["truth"] in ("straight", "out_of_range")))


def threshold_points(available):
    grouped = defaultdict(list)
    for event in available:
        grouped[event["presence_score"]].append(event)
    counts = {track: [0, 0, 0, 0] for track in reference}
    for threshold, events in sorted(grouped.items(), reverse=True):
        for event in events:
            totals = counts[event["track"]]
            for i, value in enumerate(increment(event["forecast"])):
                totals[i] += value
        yield dict(threshold=threshold, magione=counts["magione"].copy(), imola=counts["imola"].copy())


def pareto(points):
    result = []
    for point in points:
        left, right, joint, mag_false = point["magione"]
        imola_false = point["imola"][3]
        if not left:
            continue
        metric = (left, right, joint, -mag_false, -imola_false)
        if any(all(x >= y for x, y in zip(kept[0], metric, strict=True)) for kept in result):
            continue
        result = [kept for kept in result if not all(
            x >= y for x, y in zip(metric, kept[0], strict=True))]
        result.append((metric, point))
    return [point for _, point in result]


def evaluate(events, threshold):
    updates = {replay.key(event["forecast"]): event["forecast"] for event in events
               if event["presence_score"] >= threshold}
    return {track: dict(replay.compact(report), paired_with_blend=replay.matched_changes(baselines[track], report))
            for track, rows in reference.items()
            for report in [replay.reports([updates.get(replay.key(row), row) for row in rows])]}


def sustained_visual(lookup, length, aggregation):
    values = {}
    history = deque(maxlen=length)
    previous = None
    for row in geometry:
        key = replay.key(row)
        evidence = lookup.get(key)
        if evidence is None or not row["motion_contiguous"] or previous is None:
            history.clear()
        if evidence is not None:
            history.append(evidence["presence_score"])
            if len(history) == length:
                score = min(history) if aggregation == "minimum" else sum(history)/length
                values[key] = dict(evidence, presence_score=score)
        previous = key if evidence is not None else None
    return values

comparisons = []
coverage = []
outcomes = []
best = None
variants = visual.copy()
for name, rows in visual.items():
    for length in (2, 3):
        for aggregation in ("minimum", "mean"):
            variants[f"{name}_{length}_{aggregation}"] = sustained_visual(rows, length, aggregation)
for geometry_name, choices in geometry_choices.items():
    candidates = [(track, row, choices[replay.key(row)]) for track, rows in reference.items() for row in rows
                  if choices[replay.key(row)] and row["state"] not in ("left", "right")]
    for model, lookup in sorted(variants.items()):
        initial = []
        for track, row, choice in candidates:
            key = replay.key(row)
            evidence = lookup.get(key)
            if evidence is None:
                continue
            initial.append(dict(track=track, forecast=replay.rescore(row, choice),
                                presence_score=evidence["presence_score"],
                                visual_distance_m=evidence["distance_m"],
                                side_agreement=(evidence["right_minus_left"] > 0) == (choice[0] == 2),
                                distance_difference=abs(evidence["distance_m"]-choice[1])))
        coverage.append(dict(geometry=geometry_name, model=model,
              available={track: sum(event["track"] == track for event in initial) for track in reference},
              candidates={track: sum(t == track for t, _, _ in candidates) for track in reference}))
        for require_side in (False, True):
            for limit in (None, 3., 6.):
                available = [event for event in initial if (event["side_agreement"] or not require_side)
                             and (limit is None or event["distance_difference"] <= limit)]
                for distance_source in ("boundary", "mean", "ordinal"):
                    current = available
                    if distance_source != "boundary":
                        current = []
                        for event in available:
                            row = event["forecast"]
                            distance = event["visual_distance_m"]
                            if distance_source == "mean":
                                distance = .5*(distance+row["predicted_distance_m"])
                            if not 0 <= distance <= 20.:
                                continue
                            current.append(dict(event, forecast=replay.rescore(
                                row, (row["predicted_direction"], distance))))
                    points = list(threshold_points(current))
                    front = pareto(points)
                    name = dict(geometry=geometry_name, model=model, require_side=require_side,
                                distance_limit=limit, distance_source=distance_source)
                    zero_false = [p for p in points if p["magione"][0] and not p["magione"][3] and not p["imola"][3]]
                    comparisons.append(dict(**name, observed_thresholds=len(points), zero_added_false=len(zero_false),
                                            pareto=front))
                    for point in front:
                        outcomes.append(dict(**name, **point))
                        mag = point["magione"]
                        imola_false = point["imola"][3]
                        if mag[0] and mag[2]:
                            # Prefer fewer extra false alerts, then more left and joint gains.
                            rank = (mag[3]+imola_false, -mag[0], -mag[2], -mag[1])
                            if best is None or rank < best[0]:
                                best = rank, dict(**name, **point), current

best_report = dict(setting=best[1], reports=evaluate(best[2], best[1]["threshold"])) if best else None
if best:
    jsonl(out/"selected-additions.jsonl", [event for event in best[2] if event["presence_score"] >= best[1]["threshold"]])
jsonl(out/"operating-points.jsonl", outcomes)
inputs = (Path(__file__), temporal/"result.json", temporal/"replay.py", temporal/"geometry.jsonl",
          uncertainty/"result.json", uncertainty/"bounds.jsonl", factorized/"result.json",
          factorized/"predictions.jsonl", *paths.values())
write(out/"result.json", dict(baseline=baseline, comparisons=comparisons, coverage=coverage,
    lowest_false_with_joint_and_left_gain=best_report, candidate_promoted=False, seed=None,
    selected_additions_sha256=digest(out/"selected-additions.jsonl") if best else None,
    termination="development_archive_end", operating_points_sha256=digest(out/"operating-points.jsonl"),
    inputs={str(p):digest(p) for p in inputs}))
print("models", len(visual), "time_profiles", len(variants), "configurations", len(comparisons),
      "thresholds", sum(row["observed_thresholds"] for row in comparisons),
      "zero_false", sum(row["zero_added_false"] for row in comparisons))
print(best_report)
