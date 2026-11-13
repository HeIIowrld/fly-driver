from collections import Counter
from pathlib import Path
from evaluate_fused_corner import frame_key, load_score, read_rows
from train_vision_teacher_line import digest, jsonl, read, write

root=Path("runs/vision-imola-approach-diagnostic-20261111-01")
result_path=root/"result.json"
result=read(result_path)
base=read(root/"base/result.json")
assert result["base_receipt_sha256"]==digest(root/"base/result.json")
scores={
    "parent":load_score(root/"base/parent/score",base["score_receipt_sha256"]["parent"]),
    "mixed":load_score(root/"base/candidate/score",base["score_receipt_sha256"]["candidate"]),
    "candidate":load_score(root/"score",result["score_receipt_sha256"])
}
raw=root/"forecasts/rows.jsonl"
assert result["forecasts_sha256"]==digest(raw)
forecasts=read_rows(raw)
changes=[]
counts={name:Counter() for name in scores}
for parent,mixed,candidate,forecast in zip(*scores.values(),forecasts,strict=True):
    assert len({frame_key(r) for r in (parent,mixed,candidate,forecast)})==1
    label=parent["label"]
    truth=parent["truth"]
    for name,row in zip(counts,(parent,mixed,candidate),strict=True):
        if truth=="approach":
            if row["direction_hit"]: counts[name]["direction_"+label]+=1
            if row["joint_hit"]: counts[name]["joint_"+label]+=1
        if truth in ("straight","out_of_range") and row["state"] in ("left","right"):
            counts[name][truth+"_false"]+=1
    fields=("state","predicted_distance_m","direction_hit","joint_hit")
    a,b,c=[tuple(row[k] for k in fields) for row in (parent,mixed,candidate)]
    if (truth in ("straight","out_of_range","approach") and (a!=b or b!=c)) or forecast.get("restored_prior"):
        changes.append(dict(frame=parent["frame"],truth=truth,label=label,corner=parent["physical_corner"],
                            actual=parent["truth_distance_m"],parent=a,mixed=b,candidate=c,
                            restored=bool(forecast.get("restored_prior"))))
path=root/"component-rows.jsonl"
jsonl(path,changes)
approach=[r for r in changes if r["truth"]=="approach"]
write(root/"component-result.json",dict(stages=counts,changes_sha256=digest(path),
      restored_by_truth=Counter(r["truth"] for r in changes if r["restored"]),
      final_stage_left=[r for r in approach if r["label"]=="1" and r["mixed"][2]!=r["candidate"][2]],
      program_sha256=digest(Path(__file__)),source_sha256=digest(result_path),
      termination="frozen_imola_component_comparison_complete"))
print(counts)
