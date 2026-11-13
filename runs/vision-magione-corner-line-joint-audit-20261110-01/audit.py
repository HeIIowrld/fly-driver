import json
from collections import Counter
from pathlib import Path

from train_vision_teacher_line import digest, jsonl, read, write

root = Path("runs/vision-frozen-next-corner-20261110-01")
out = Path(__file__).parent
sources = {
    "labels":Path("runs/vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl"),
    "forecast":root/"archive-inference/rows.jsonl",
    "line":root/"archive-native-direct-first/rows.jsonl",
}
indices = {}
for name, path in sources.items():
    proof = read(path.with_name("result.json"))
    assert digest(path) == proof["rows_sha256"]
    rows = list(map(json.loads,path.read_text(encoding="utf8").splitlines()))
    indices[name] = {(row["run"],row["frame"]):row for row in rows}
    assert len(rows) == len(indices[name])
assert indices["labels"].keys() == indices["forecast"].keys() == indices["line"].keys()

summary, by_direction, examples = Counter(), {1:Counter(),2:Counter()}, []
for key, label in indices["labels"].items():
    forecast = indices["forecast"][key]
    planned = indices["line"][key]
    target = label["distance"]
    is_approach = label["direction"] in (1,2) and target is not None and 4 <= target <= 20
    if not is_approach:
        summary["outside_near_band"] += 1
        summary["out_of_band_alerts"] += forecast["alert"]
        continue
    side = by_direction[label["direction"]]
    side["approach"] += 1
    detected = forecast["alert"] and forecast["direction"] == label["direction"]
    side["direction_hits"] += detected
    distance_ok = detected and abs(forecast["distance_m"]-target) <= 5
    side["direction_and_distance_hits"] += distance_ok
    line = planned["line"]
    horizon = None if line is None else line["arc_m"][-1]
    reaches_onset = horizon is not None and horizon >= target
    side["planned_path"] += line is not None
    side["joint_direction_distance_and_path"] += distance_ok and line is not None
    side["joint_direction_distance_and_path_reaches_onset"] += distance_ok and reaches_onset
    examples.append(dict(run=key[0],frame=key[1],actual_direction=label["direction"],
        actual_onset_m=target,forecast_direction=forecast["direction"],forecast_onset_m=forecast["distance_m"],
        alert=forecast["alert"],line_horizon_m=horizon,
        path_reaches_approximate_onset=reaches_onset,line=line,
        joint_hit=distance_ok and reaches_onset, status=planned["status"]))
jsonl(out/"approach-rows.jsonl",examples)
write(out/"result.json",dict(status="completed_archived_corner_and_line_audit",
    outside=dict(summary),by_direction={str(k):dict(v) for k,v in by_direction.items()},
    total=dict(sum(by_direction.values(),Counter())), scope="Magione development recordings, no driving",
    line_reach_is_approximate=True, apex_verified=False, geometry_evaluation_only=True,
    hashes={str(p):digest(p) for p in (*sources.values(),Path(__file__))},
    rows_sha256=digest(out/"approach-rows.jsonl"),seed=None,termination="archive_end"))
print(summary,by_direction)
