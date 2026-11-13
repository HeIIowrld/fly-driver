"""Test causal one- and two-capture retention of front bend evidence."""
import json
from pathlib import Path
import numpy as np
from train_vision_teacher_line import digest,jsonl,read,write

out=Path(__file__).parent
source=out/"rows.jsonl"
proof=read(out/"result.json")
assert digest(source)==proof["rows_sha256"]
root=out.parent
a=root/"vision-monza-two-lap-presence-20261111-01/rows.jsonl"
b=root/"vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl"
assert digest(a)==read(a.with_name("flyvis-result.json"))["rows_sha256"]
assert digest(b)==read(b.with_name("result.json"))["rows_sha256"]
time={}
for path in (a,b):
    time.update({(r["run"],r["frame"]):r["time"] for r in map(json.loads,path.read_text().splitlines())})
rows=list(map(json.loads,source.read_text().splitlines()))
indexed={(r["run"],r["frame"]):r for r in rows}
for r in rows:
    run,frame=r["run"],r["frame"]
    previous=[r]
    for lag in (1,2):
        key=run,frame-lag
        p=indexed.get(key)
        if p is None or p["cohort"]!=r["cohort"] or not 0<time[run,frame]-time[key]<=.24:
            break
        previous.append(p)
    r["peak_1"]=max(p["peak"] for p in previous[:2])
    r["peak_2"]=max(p["peak"] for p in previous)
train=[r for r in rows if r["cohort"]=="monza_train_oof"]
def counts(cohort, accepted):
    hits=list(zip(cohort,accepted,strict=True))
    return dict(left=sum(a and r["near"] and r["actual"]==1 for r,a in hits),
                right=sum(a and r["near"] and r["actual"]==2 for r,a in hits),
                outside=sum(a and not r["near"] for r,a in hits),
                straight=sum(a and r["previous_straight"] for r,a in hits))
def classify(cohort,mode,angle,cut):
    return [bool(r["eligible"] and r[mode]>=angle and r["score"]>=cut) for r in cohort]
base_cut=read(root/"vision-monza-presence-temporal-ablation-20261111-01/result.json")["summary"]["latest_and_changes"]["threshold"]
summary={}
for mode in ("peak","peak_1","peak_2"):
    summary[mode]={}
    for angle in (.025,.05,.10):
        cuts={r["score"] for r in train if r[mode]>=angle}
        cuts.add(float(np.nextafter(max(r["score"] for r in train),np.inf)))
        trials=[]
        for cut in cuts:
            score=counts(train,classify(train,mode,angle,cut))
            if score["outside"]<=23:
                trials.append(((min(score["left"]/18,score["right"]/34),score["left"]+score["right"],
                                -score["outside"],cut),cut))
        _,cut=max(trials)
        summary[mode][str(angle)]={"cut":cut}
        for name in ("monza_train_oof","monza_validation","magione_development"):
            cohort=[r for r in rows if r["cohort"]==name]
            summary[mode][str(angle)][name]=dict(
                same_cut=counts(cohort,classify(cohort,mode,angle,base_cut)),
                refit_cut=counts(cohort,classify(cohort,mode,angle,cut)))
jsonl(out/"causal-rows.jsonl",rows)
write(out/"causal-result.json",dict(summary=summary,seed=None,termination="archive_end",
    revision=proof["revision"],rows_sha256=digest(out/"causal-rows.jsonl"),
    hashes={str(p):digest(p) for p in (source,a,b,Path(__file__))}))
for mode,r in summary.items():
    print(mode,r["0.05"])
