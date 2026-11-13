from pathlib import Path
from collections import Counter
from score_corner_captures import summarize
from evaluate_fused_corner import load_score
from train_vision_teacher_line import digest, read, write

root=Path("runs/vision-imola-approach-diagnostic-20261111-01")
result_path=root/"result.json"
result=read(result_path)
assert result["status"]=="evaluated"
base=read(root/"base/result.json")
assert result["base_receipt_sha256"]==digest(root/"base/result.json")
inputs={"parent":(root/"base/parent/score",base["score_receipt_sha256"]["parent"]),
        "recovered":(root/"score",result["score_receipt_sha256"])}
reports={}
for name,(folder,checksum) in inputs.items():
    rows=load_score(folder,checksum)
    full=summarize(rows)
    reports[name]={"by_direction":full["by_direction"],"outputs":full["outputs"],
       "false":{k:v["frames"] for k,v in full["false_alerts"].items()},
       "stable":[dict(corner=e["physical_corner"],side=e["label"],first_lead_m=e["first_stable_lead_m"])
                 for e in full["events"] if e["first_stable_frame"] is not None],
       "distance_mae_m":full["distance_mae_m"],
       "distance_change":{k:v for k,v in full["distance_change"].items() if k!="rows"},
       "missed":[(*k,v) for k,v in Counter((r["label"],r["state"],r["abstain_reason"])
                for r in rows if r["truth"]=="approach" and not r["direction_hit"]).items()]}
write(root/"summary.json",dict(reports=reports,source_sha256={str(result_path):digest(result_path)},
      termination="frozen_imola_diagnostic_summarized"))
print(reports)
