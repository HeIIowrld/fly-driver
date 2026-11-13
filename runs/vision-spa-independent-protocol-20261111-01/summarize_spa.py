"""Summarize the three verified Spa prediction timelines."""
from pathlib import Path
from collections import Counter

from evaluate_fused_corner import load_score
from score_corner_captures import summarize
from train_vision_teacher_line import digest, read, write

root = Path("runs/vision-spa-independent-evaluation-20261111-01")
result_path = root/"result.json"
result = read(result_path)
assert result["status"] == "evaluated"
full = root/"comparison"
assert digest(full/"result.json") == result["primary_receipt_sha256"]
assert digest(full/"base/result.json") == result["secondary_receipt_sha256"]
assert digest(root/"admission/result.json") == result["admission_sha256"]
primary = read(full/"result.json")
base = read(full/"base/result.json")
admission = read(root/"admission/result.json")
sources = {
    "parent": (full/"base/parent/score", base["score_receipt_sha256"]["parent"]),
    "mixed": (full/"base/candidate/score", base["score_receipt_sha256"]["candidate"]),
    "recovered": (full/"score", primary["score_receipt_sha256"]),
}
reports = {}
for name, (folder, checksum) in sources.items():
    rows = load_score(folder, checksum)
    stats = summarize(rows)
    stable = [dict(corner=e["physical_corner"], side=e["label"], first_lead_m=e["first_stable_lead_m"])
              for e in stats["events"] if e["first_stable_frame"] is not None]
    reports[name] = dict(by_direction=stats["by_direction"], outputs=stats["outputs"],
        false={k:v["frames"] for k,v in stats["false_alerts"].items()},
        stable=stable, stable_corners=len({e["corner"] for e in stable}),
        distance_mae_m=stats["distance_mae_m"],
        distance_change={k:v for k,v in stats["distance_change"].items() if k!="rows"},
        missed=[(*k,v) for k,v in Counter((r["label"],r["state"],r["abstain_reason"])
                 for r in rows if r["truth"]=="approach" and not r["direction_hit"]).items()])
assert not (root/"summary.json").exists()
write(root/"summary.json",dict(reports=reports,source_sha256={str(result_path):digest(result_path)},
     termination="frozen_spa_diagnostic_summarized"))
print(reports)
