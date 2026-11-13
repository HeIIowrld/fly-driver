"""Compare frozen readout failures by physical corner and test a train-only margin."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from infer_frozen_next_corner import load_readout
from train_vision_teacher_line import digest, jsonl, read, write

out=Path(__file__).parent
root=out.parent
data=root/"vision-monza-two-lap-presence-20261111-01"
source=root/"vision-monza-corner-guided-replay-20261111-01/forecasts.jsonl"
original=root/"vision-monza-midpoint-raw-head-20261110-01/rows.jsonl"
pins=root/"vision-frozen-next-corner-20261110-01/pins.json"
magione=root/"vision-magione-left-readout-stages-20261111-01/rows.jsonl"
features=data/"features.npz"
receipt=read(data/"flyvis-result.json")
proof=read(source.with_name("forecasts-result.json"))
assert digest(source)==proof["rows_sha256"]
assert digest(features)==receipt["features_sha256"] and digest(data/"rows.jsonl")==receipt["rows_sha256"]
assert digest(original) in receipt["inputs_sha256"].values()
assert digest(pins) in proof["hashes"].values()
assert digest(magione)==read(magione.with_name("result.json"))["rows_sha256"]

def load(path):
    with path.open(encoding="utf8") as stream:
        return list(map(json.loads,stream))

fold=load(data/"rows.jsonl")
stored=load(source)
assert [r["label"] for r in stored]==fold
training_groups={r["physical_group"] for r in load(original) if r["split"]=="train"}
validation_groups={r["physical_group"] for r in fold if r["split"]=="validation"}
assert not validation_groups & training_groups
predict,threshold=load_readout(read(pins))
with np.load(features,allow_pickle=False) as packed:
    direct,reflected,available=(packed[k] for k in ("direct","reflected","reflected_available"))
scores=np.empty((len(fold),3))
distances=np.empty(len(fold))
presence=np.empty(len(fold))
# Match the reference's one-row arithmetic at the frozen threshold.
for i in range(len(fold)):
    ss,dd,pp=predict(direct[i:i+1],reflected[i:i+1] if available[i] else None,return_scores=True)
    scores[i],distances[i],presence[i]=ss[0],dd[0],pp[0]


rows=[]
for i,(label,record) in enumerate(zip(fold,stored,strict=True)):
    actual=label["direction"]
    distance=label["distance"]
    raw=int(np.argmax(scores[i])) if np.count_nonzero(scores[i]==scores[i].max())==1 else 0
    present=bool(presence[i]>=threshold)
    in_range=bool(distances[i]<=20.)
    alert=bool(raw and present and in_range)
    assert raw==record["base"]["direction"] and alert==record["base"]["alert"]
    assert abs(presence[i]-record["base"]["presence_score"])<1e-10
    rows.append(dict(label,actual=actual,near=bool(actual and distance is not None and 4<=distance<=20),
        raw=raw,present=present,in_range=in_range,alert=alert,scores=scores[i].tolist(),
        reflected_available=bool(available[i])))

def summarize(group, predictions):
    counts=Counter()
    for r,choice in zip(group,predictions,strict=True):
        if r["near"]:
            side=r["actual"]
            counts[f"near_{side}"]+=1
            counts[f"hit_{side}"]+=choice==side
        elif choice:
            counts["outside_alerts"]+=1
    return dict(counts)

def base(group):
    return [r["raw"] if r["alert"] else 0 for r in group]

def alternate(group, margin):
    selected=[]
    for r in group:
        if r["alert"]:
            selected.append(r["raw"])
            continue
        left,right=r["scores"][1:]
        can_rescue=r["raw"]==0 and r["present"] and r["in_range"] and abs(left-right)>margin
        selected.append((1 if left>right else 2) if can_rescue else 0)
    return selected

train=[r for r in rows if r["split"]=="train"]
valid=[r for r in rows if r["split"]=="validation"]
baseline=summarize(train,base(train))
options=[]
for margin in (0.,.025,.05,.075,.10,.15,.20,.30,.50):
    metrics=summarize(train,alternate(train,margin))
    if metrics.get("outside_alerts",0)<=baseline.get("outside_alerts",0):
        rate=sum(metrics[f"hit_{side}"]/metrics[f"near_{side}"] for side in (1,2))/2
        options.append((rate,margin,metrics))
chosen=max(options,key=lambda v:(v[0],v[1])) if options else None
groups={}
for split in ("train","validation"):
    relevant=[r for r in rows if r["split"]==split and r["near"]]
    for group in sorted({r["physical_group"] for r in relevant}):
        for side in (1,2):
            event=[r for r in relevant if r["physical_group"]==group and r["actual"]==side]
            if event:
                groups[f"{split}/{group}/{side}"]=dict(n=len(event),events=sorted({r["event"] for r in event}),
                    raw=dict(Counter(r["raw"] for r in event)), present=sum(r["present"] for r in event),
                    in_range=sum(r["in_range"] for r in event),alerts=dict(Counter(base(event))),
                    observed_laps=len({r["run"] for r in event}))
summary={}
for name,group in (("monza_train",train),("monza_validation",valid),("magione_development",load(magione))):
    summary[name]=dict(baseline=summarize(group,base(group)),
                       no_margin=summarize(group,alternate(group,0.)),
                       train_selected=summarize(group,alternate(group,chosen[1])) if chosen else None)
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(summary=summary,by_physical_group=groups,
    chosen_margin=chosen[1] if chosen else None,seed=None,termination="archive_end",revision=proof["revision"],
    hashes={str(path):digest(path) for path in (source,original,data/"rows.jsonl",features,pins,magione,Path(__file__))},
    rows_sha256=digest(out/"rows.jsonl")))
print(summary)
print(groups)
