from collections import Counter
from pathlib import Path
from score_corner_captures import adjacent, summarize
from train_vision_teacher_line import digest, jsonl, read, write
from evaluate_fused_corner import frame_key, read_rows
from evaluate_independent_corner import compare

root = Path("runs/vision-mugello-independent-evaluation-20261111-01")
protocol = Path("runs/vision-corner-recovered-runner-20261111-01/mugello-base-protocol.json")
rules = read(protocol)["comparison"]
fusion_path = Path("runs/vision-corner-fusion-monza-transfer-20261111-01/development-pins.json")
fusion = read(fusion_path)
out = Path(__file__).parent
audit = out/"rows.jsonl"
candidate = root/"score/rows.jsonl"
parent = root/"base/parent/score/rows.jsonl"
observations = read_rows(audit)
rows = read_rows(candidate)
cases = {}
all_rows = []
for method in ("no_background", "two_observations", "two_observations_with_distance", "boundary_motion"):
    previous = None
    changed = Counter()
    new_rows = []
    for obs, row in zip(observations, rows, strict=True):
        assert frame_key(obs) == frame_key(row)
        m = obs["mixed"]
        side = m["side_without_background"] if m and m["presence_pass"] and m["distance_pass"] else None
        distance = m["distance_m"] if side else None
        if method == "boundary_motion":
            distance = obs["boundary_distance_m"]
            side = obs["boundary_side"] if distance is not None and 0 <= distance <= 20 else None
        current = (row, side, distance)
        continuous = previous is not None and adjacent(previous[0], row) and side == previous[1]
        delta = obs["travel_m"]
        consistent = (continuous and side is not None and delta is not None and delta > 0
                      and distance < previous[2])
        allow = (method == "no_background" or method == "two_observations" and continuous
                 or method == "two_observations_with_distance" and consistent)
        if method == "boundary_motion":
            allow = (continuous and side is not None and delta is not None
                     and delta >= fusion["minimum_local_travel_m"]
                     and abs(distance-(previous[2]-delta)) <= fusion["maximum_position_innovation_m"])
        previous = current
        if row["state"] == "straight" and side and allow:
            label = "left" if side == 1 else "right"
            is_hit = row["truth"] == "approach" and row["label"] == str(side)
            errors = [distance - truth for truth in row["truth_endpoints_m"]] if is_hit else []
            change = dict(state=label, predicted_distance_m=distance, predicted_direction=side,
                          direction_hit=is_hit, joint_hit=bool(errors) and max(map(abs, errors)) <= 5,
                          signed_distance_error_m=sum(errors)/len(errors) if errors else None,
                          absolute_distance_error_m=sum(map(abs, errors))/len(errors) if errors else None)
            row = dict(row, **change)
            changed[(row["truth"],label,row["direction_hit"],row["joint_hit"])] += 1
            all_rows.append(dict(row, method=method))
        new_rows.append(row)
    stable = summarize(new_rows)["events"]
    track = rows[0]["run"]
    cases[method] = dict(comparison=compare({"parent":{track:read_rows(parent)},
                "candidate":{track:new_rows}}, rules),
                stable=[(e["physical_corner"],e["first_stable_lead_m"]) for e in stable if e["first_stable_frame"] is not None],
                changes=[(*k,v) for k,v in changed.items()])
jsonl(out/"counterfactual-changes.jsonl",all_rows)
write(out/"counterfactual-result.json",dict(cases=cases, revision=read(out/"result.json")["revision"], seed=None,
      changes_sha256=digest(out/"counterfactual-changes.jsonl"),
      hashes={str(p):digest(p) for p in (protocol,fusion_path,audit,candidate,parent,Path(__file__))},
      termination="post_evaluation_counterfactual_only"))
for key, v in cases.items():
    print(key,v["comparison"]["metrics"]["candidate"],v["stable"],v["changes"])
