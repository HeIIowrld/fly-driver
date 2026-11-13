"""Test causal distance corrections using only consecutive frozen visual predictions."""
from collections import defaultdict, deque
import json
from pathlib import Path
from statistics import mean

from evaluate_independent_corner import compare
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
root = out.parent
inputs = [Path(__file__), Path("scripts/evaluate_independent_corner.py"), Path("scripts/score_corner_captures.py")]


def load(name):
    source = root/name
    proof = read(source/"result.json")
    assert digest(source/"rows.jsonl") == proof["rows_sha256"]
    inputs.extend((source/"rows.jsonl", source/"result.json"))
    return list(map(json.loads, (source/"rows.jsonl").read_text().splitlines())), proof["revision"]


parent, _ = load("vision-corner-common-magione-20261111-04")
candidate, revision = load("vision-corner-blended-common-20261111-01")
protocol = root/"vision-corner-independent-protocol-20261111-01/protocol.json"
inputs.append(protocol)
rules = read(protocol)["comparison"]


def groups(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["run"]].append(row)
    return result


def corrected(rows, cap):
    history = defaultdict(lambda: deque(maxlen=3))
    previous, filtered = {}, {}
    for row in rows:
        run = row["run"]
        raw = row["predicted_distance_m"]
        old = previous.get(run)
        output = dict(row)
        if row["state"] not in ("left", "right"):
            previous.pop(run, None)
            history[run].clear()
            yield output
            continue
        if not old or not adjacent(old, row) or old["state"] != row["state"]:
            history[run].clear()
        elif raw > filtered[run] and history[run]:
            drift = mean(history[run])
            if drift > 0:
                output["predicted_distance_m"] = max(raw-cap, filtered[run]-drift, 0)
        if old and adjacent(old, row) and old["state"] == row["state"]:
            history[run].append(max(0, old["predicted_distance_m"]-raw))
        previous[run] = row
        filtered[run] = output["predicted_distance_m"]
        if row["direction_hit"]:
            errors = [filtered[run]-truth for truth in row["truth_endpoints_m"]]
            output.update(joint_hit=max(map(abs, errors)) <= 5,
                          signed_distance_error_m=mean(errors), absolute_distance_error_m=mean(map(abs,errors)))
        yield output


reports, details = {}, []
for cap in (0., .025, .05, .1, .2, .4):
    rows = list(corrected(candidate, cap))
    reports[str(cap)] = dict(
        vs_parent=compare(dict(parent=groups(parent), candidate=groups(rows)), rules),
        vs_raw=compare(dict(parent=groups(candidate), candidate=groups(rows)), rules))
    details.extend(dict(cap_m=cap,run=r["run"],frame=r["frame"],distance_m=r["predicted_distance_m"])
                   for r in rows if r["state"] in ("left","right"))
    print(cap,reports[str(cap)]["vs_raw"]["metrics"]["candidate"]["joint"],
          reports[str(cap)]["vs_raw"]["distances"]["candidate"],
          reports[str(cap)]["vs_parent"]["distances"]["candidate"])
jsonl(out/"rows.jsonl",details)
write(out/"result.json",dict(reports=reports, seed=None, revision=revision, candidate_promoted=False,
      actor_inputs=["prior frozen distance predictions", "capture order and timing"],
      hashes={str(p):digest(p) for p in inputs}, rows_sha256=digest(out/"rows.jsonl"),
      termination="development_archive_end"))
