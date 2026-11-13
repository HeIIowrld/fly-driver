"""Verify frozen evaluation receipts and locate missed corner decisions."""
from collections import Counter
import json
from pathlib import Path

from train_vision_teacher_line import digest,read,write

out=Path(__file__).parent
capture=out.parent/"vision-laguna-independent-capture-20261111-01"
protocol=out.parent/"vision-corner-independent-track-options-20261111-01/ks_laguna_seca-protocol.json"
result=read(out/"result.json")
assert result["protocol_sha256"]==digest(protocol)
assert digest(out/"admission-rows.jsonl")==result["admission_rows_sha256"]
assert read(capture/"result.json")["archive"]["bundle_sha256"]==result["camera_bundle_sha256_by_run"]["laguna-seca-ai-01"]
inputs=[Path(__file__),out/"result.json",out/"admission-rows.jsonl",capture/"result.json",protocol]
summary={}
for name in ("parent","candidate"):
    folder=out/name/"score"
    receipt=folder/"result.json"
    trace=folder/"rows.jsonl"
    assert result["score_receipt_sha256"][name]==digest(receipt)
    assert read(receipt)["rows_sha256"]==digest(trace)
    inputs.extend((receipt,trace))
    counts={"1":Counter(),"2":Counter()}
    for row in map(json.loads,trace.read_text(encoding="utf8").splitlines()):
        if row["truth"]!="approach":
            continue
        label=row["label"]
        direction=row["predicted_direction"]
        if row["direction_hit"]:
            stage="direction_hit"
        elif row["state"]=="no_output" or direction is None:
            stage="unavailable"
        elif direction==0:
            stage="raw_background"
        elif str(direction)!=label:
            stage="opposite_raw_direction"
        else:
            stage=row["abstain_reason"] or "unexplained"
        counts[label][stage]+=1
    assert sum(counts["1"].values())==38 and sum(counts["2"].values())==25
    summary[name]=counts
write(out/"failure-stages.json",dict(revision=result["revision"],seed=None,summary=summary,
    hashes={str(p):digest(p) for p in inputs},termination="frozen_holdout_diagnosis_end"))
print(summary)
