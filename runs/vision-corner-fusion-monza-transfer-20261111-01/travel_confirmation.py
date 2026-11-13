"""Compare elapsed vehicle travel with a fixed frame count for geometric confirmation."""
import importlib.util
from pathlib import Path

from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
runs = out.parent
temporal = runs/"vision-corner-temporal-boundaries-20261111-01"
uncertainty = runs/"vision-corner-boundary-uncertainty-20261111-01"
factor = runs/"vision-corner-factorized-20261111-01"
fused = runs/"vision-corner-fused-confirmation-20261111-01"
spec = importlib.util.spec_from_file_location("temporal_replay", temporal/"replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
selection = read(fused/"result.json")["lowest_false_with_joint_and_left_gain"]["setting"]
assert selection["model"] == "contrast-10000" and selection["distance_source"] == "ordinal"
assert selection["geometry"] == "whole_shape_min_0.002" and not selection["require_side"]
assert selection["distance_limit"] is None
base = read(temporal/"result.json")
unc = read(uncertainty/"result.json")
mono = read(out/"result.json")
bounds = replay.read_rows(uncertainty/"bounds.jsonl", unc["bounds_sha256"])
geometry = [dict(r, curvature={"history_0.24": r["curvature"].get("whole_shape_min", [])}) for r in bounds]
monza_geometry = [r["shape"] for r in replay.read_rows(out/"geometry.jsonl", mono["geometry_sha256"])]
score_paths = {
    "magione": runs/"vision-corner-independent-runner-rehearsal-20261111-01"/"replay"/"candidate"/"score"/"rows.jsonl",
    "imola": runs/"vision-corner-imola-full-blended-20261111-01"/"rows.jsonl",
}
references = {track: replay.read_rows(path, base["scoring_inputs"][str(path)]) for track, path in score_paths.items()}
references["monza"] = [row["parent"] for row in replay.read_rows(out/"rows.jsonl", mono["rows_sha256"])]
scores = {replay.key(row): row for row in replay.read_rows(factor/"predictions.jsonl", read(factor/"result.json")["prediction_sha256"])
          if row["model"] == selection["model"] and row["cohort"] in (
              "monza_validation", "magione_transfer", "imola_negative_transfer")}


def confirm_by_travel(source, minimum, tolerance):
    prior = None
    travel_total = 0.
    choices = {}
    for row in source:
        choice = replay.decide(row["curvature"].get("history_0.24", []), .002)
        travel = row["travel_m"]
        if (travel is not None and travel >= .10 and choice and choice[0] and prior and prior[0] == choice[0] and
                abs(choice[1]-(prior[1]-travel)) <= tolerance):
            travel_total += travel
        else:
            travel_total = 0.
        choices[replay.key(row)] = choice if travel_total >= minimum else None
        prior = choice
    return choices


result = {}
distance_comparisons = {}
for track, rows in references.items():
    source = monza_geometry if track == "monza" else geometry
    variants = {"three_frames": replay.confirmed_predictions(source, .002, 3, 1.)}
    for minimum in (.75, 1.):
        for tolerance in (1., 2.):
            variants[f"{minimum}_{tolerance}"] = confirm_by_travel(source, minimum, tolerance)
    baseline = replay.reports(rows)
    summary = {"existing_blend": replay.compact(baseline)}
    for name, choices in variants.items():
        updates = []
        stage_counts = {"geometry_without_baseline": 0, "added": 0, "added_left": 0}
        for row in rows:
            key = replay.key(row)
            choice, evidence = choices[key], scores.get(key)
            if choice and row["state"] not in ("left", "right"):
                stage_counts["geometry_without_baseline"] += 1
                if (evidence and evidence["presence_score"] >= selection["threshold"] and
                        0 <= evidence["distance_m"] <= 20.):
                    row = replay.rescore(row, (choice[0], evidence["distance_m"]))
                    stage_counts["added"] += 1
                    stage_counts["added_left"] += int(row["direction_hit"] and row["label"] == "1")
            updates.append(row)
        report = replay.reports(updates)
        summary[name] = dict(replay.compact(report), stages=stage_counts,
                             paired_with_blend=replay.matched_changes(baseline, report))
        if name != "0.75_1.0":
            continue
        travel_at = {replay.key(r): r["travel_m"] for r in source}
        distances = {}
        for cap in (.05, .25, .5, 1.):
            adjusted = []
            previous = None
            for row in updates:
                travel = travel_at[replay.key(row)]
                if row["state"] in ("left", "right"):
                    raw = row["predicted_distance_m"]
                    if (previous is not None and travel is not None and travel > 0 and
                            previous["run"] == row["run"] and previous["frame"]+1 == row["frame"] and
                            previous["state"] == row["state"]):
                        predicted = max(0., previous["predicted_distance_m"]-travel)
                        if raw > predicted:
                            revised = max(raw-cap, predicted)
                            row = replay.rescore(row, (row["predicted_direction"], revised))
                    previous = row
                else:
                    previous = None
                adjusted.append(row)
            adjusted_report = replay.reports(adjusted)
            distances[str(cap)] = dict(replay.compact(adjusted_report),
                paired_with_travel=replay.matched_changes(report, adjusted_report),
                paired_with_blend=replay.matched_changes(baseline, adjusted_report))
        distance_comparisons[track] = distances
    result[track] = summary
write(out/"travel-confirmation.json", dict(summary=result, distance_comparisons=distance_comparisons, seed=None, termination="development_archive_end",
      candidate_promoted=False, inputs={str(p):digest(p) for p in (Path(__file__), temporal/"replay.py",
          uncertainty/"result.json", uncertainty/"bounds.jsonl", out/"result.json", out/"geometry.jsonl",
          out/"rows.jsonl", factor/"predictions.jsonl", fused/"result.json", *score_paths.values())}))
for track, summary in result.items():
    for name, row in summary.items():
        print(track, name, *(row[k] for k in ("left", "right", "joint", "outside")), row.get("stages"))


for track, variants in distance_comparisons.items():
    for cap, r in variants.items():
        print("distance", track, cap, r["joint"], r["stable_leads_m"], r["paired_with_blend"])
