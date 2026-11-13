from argparse import Namespace
from pathlib import Path
import json

from evaluate_independent_corner import adjust_forecasts, compare
from evaluate_live_corner_geometry import evaluate
from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
root = out.parent
prior = root/"vision-corner-independent-protocol-20261111-01/protocol.json"
study = root/"vision-corner-visual-distance-20261111-01/result.json"
protocol = read(prior)
for path, expected in protocol["hashes"].items():
    assert digest(Path(path)) == expected
protocol["comparison"]["candidate_visual_distance"] = dict(cap_m=.05, history_intervals=3)
protocol["supersedes_protocol_sha256"] = digest(prior)
protocol["distance_selection"] = "Largest tested cap preserving Magione joint hits"
protocol["hashes"].update({str(p): digest(p) for p in
                           (prior, study, Path(__file__), Path("scripts/evaluate_independent_corner.py"))})
write(out/"protocol.json", protocol)

replay = root/"vision-corner-independent-runner-rehearsal-20261111-01/replay"
receipt = read(replay/"result.json")
assert receipt["status"] == "evaluated" and receipt["track"] == "magione"
admission = replay/"admission-rows.jsonl"
assert digest(admission) == receipt["admission_rows_sha256"]
captures = list(map(json.loads, admission.read_text(encoding="utf8").splitlines()))
adjust_forecasts(replay/"candidate/forecasts", out/"development-adjusted-forecasts",
                 captures, out/"protocol.json", .05)
archives = root/"vision-frozen-next-corner-20261110-01/archives.json"
evaluate(Namespace(output=out/"development-score", track="magione",
    track_dir=Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks/magione"),
    common=True, pins=Path(protocol["comparison"]["candidate_pins"]),
    forecasts=out/"development-adjusted-forecasts/rows.jsonl", archive_map=archives, recordings=[]))
timelines = {}
for name, directory in (("parent", replay/"parent/score"), ("candidate", out/"development-score")):
    scores = read(directory/"result.json")
    assert scores["camera_bundle_sha256_by_run"] == receipt["camera_bundle_sha256_by_run"]
    assert digest(directory/"rows.jsonl") == scores["rows_sha256"]
    assert scores["capture_frames"] == len(captures) == 1887
    rows = list(map(json.loads, (directory/"rows.jsonl").read_text(encoding="utf8").splitlines()))
    timelines[name] = {tag: [r for r in rows if r["run"] == tag] for tag in read(archives)}
summary = compare(timelines, protocol["comparison"])
assert summary["metrics"]["candidate"]["joint"] == 51
assert summary["gates"]["distance_change"] and summary["gates"]["receding"]
assert not summary["gates"]["outside"]
write(out/"development-verification.json", dict(status="verified_development_replay",
      independent_evaluation=False, summary=summary, protocol_sha256=digest(out/"protocol.json"),
      parent_score_sha256=digest(replay/"parent/score/result.json"),
      adjusted_score_sha256=digest(out/"development-score/result.json"),
      revision=protocol["revision"], seed=None, termination="rehearsal_end"))
print(summary)
