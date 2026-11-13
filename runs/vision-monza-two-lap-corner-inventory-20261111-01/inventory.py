from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from train_next_corner import BASE,MONZA,assign,CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at,digest,jsonl,read,write

out=Path(__file__).parent
original=Path("runs/vision-next-corner-20261110-03")
teacher.CORNER=CURVATURE_THRESHOLD
base=Path("runs/vision-midpoint-aligned-head-20261110-01")
source=(base/"probe.py").read_text(encoding="utf-8-sig")
segment=source[source.index("mapfile="):source.index("all_rows=rows")]
prior=Path("runs/vision-corner-label-geometry-20261110-01/probe.py")
code=prior.read_text(encoding="utf-8-sig")
exec(code[code.index("def periodic_line("):code.index("def near(")])
metadata=read(original/"dataset.json")
exec(segment)
validation_groups=read(base/"result.json")["validation_groups"]
selected=[]
reports={}
for run,sub in MONZA:
    rows,provenance=teacher.examples(BASE/run,BASE/run/sub,teacher.road_model(mapfile),False,out)
    candidate=Counter()
    for r in rows:
        residual,j=mt.query(r["xy"])
        station=float(mg[j])
        result=assign(station,ms,ml,True)
        if residual>7 or result is None:
            continue
        event,sign,distance=result
        group=int(groups[event])
        indexes=(np.arange(station-15.,station+min(distance,30.)+1).astype(int))%len(centers)
        if group in held or np.any(np.abs((centers[indexes,None]-holdouts+length/2)%length-length/2)<=90):
            continue
        split="validation" if group in validation_groups else "train"
        if sign and distance<=30:
            band="4-20" if 4<=distance<=20 else "other_to_30"
            candidate[(split,group,sign,band)]+=1
        selected.append(dict(run=run,frame=r["index"],time=r["t"],history=r["history"],
                             split=split,physical_group=group,event=event,direction=sign,
                             distance=float(distance) if sign else None))
    reports[run]=dict(provenance=provenance,support=[dict(split=k[0],group=k[1],direction=k[2],
        band=k[3],frames=n) for k,n in sorted(candidate.items())])
    print(run,sum(candidate.values()),flush=True)
old=Path("runs/vision-monza-midpoint-raw-head-20261110-01/rows.jsonl")
assert digest(old)==read(old.with_name("result.json"))["rows_sha256"]
import json
expected=[json.loads(l) for l in old.read_text(encoding="utf8").splitlines()]
assert [r for r in selected if r["run"]==MONZA[0][0]]==expected
train={r["physical_group"] for r in selected if r["split"]=="train"}
validation={r["physical_group"] for r in selected if r["split"]=="validation"}
assert train.isdisjoint(validation)
removed=[]
for run,_ in MONZA:
    validation_frames={f for r in selected if r["run"]==run and r["split"]=="validation" for f in r["history"]}
    crossing=[r for r in selected if r["run"]==run and r["split"]=="train"
              and not validation_frames.isdisjoint(r["history"])]
    removed.extend(crossing)
    selected=[r for r in selected if r not in crossing]
    assert validation_frames.isdisjoint(f for r in selected if r["run"]==run and r["split"]=="train"
                                        for f in r["history"])
    reports[run]["near_after_history_purge"]=sum(r["run"]==run and r["direction"] and r["distance"]<=30
                                               for r in selected)
print("removed overlapping histories",[(r["run"],r["frame"],r["distance"]) for r in removed],flush=True)
jsonl(out/"removed-history.jsonl",removed)
jsonl(out/"rows.jsonl",selected)
write(out/"result.json",dict(revision=read(base/"result.json")["revision"],seed=None,reports=reports,
    n=len(selected),removed_history_sha256=digest(out/"removed-history.jsonl"),rows_sha256=digest(out/"rows.jsonl"),validation_groups=validation_groups,
    termination="archive_end",hashes={str(p):digest(p) for p in
        (old,prior,mapfile,base/"probe.py",base/"result.json",original/"dataset.json",Path(__file__))}))