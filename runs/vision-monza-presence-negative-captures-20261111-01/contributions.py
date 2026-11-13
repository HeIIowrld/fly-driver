"""Attribute the held physical group's presence score to view and time inputs."""
import ast
import json
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.linalg import solve
from flybrain_drive.flyvis_camera import VIEW_ORDER
from infer_frozen_next_corner import contrast
from train_vision_teacher_line import digest,jsonl,predict,read,write

out=Path(__file__).parent
root=out.parent
previous=root/"vision-monza-presence-temporal-ablation-20261111-01"
selected=root/"vision-monza-two-lap-presence-20261111-01"
receipt=read(selected/"flyvis-result.json")
data=selected/"features.npz"
index=selected/"rows.jsonl"
assert digest(data)==receipt["features_sha256"] and digest(index)==receipt["rows_sha256"]
meta=list(map(json.loads,index.read_text().splitlines()))
group=np.array([r["physical_group"] for r in meta])
truth=np.array([r["direction"] if r["direction"] and 4<=r["distance"]<=20 else 0 for r in meta])
train=np.array([r["split"]=="train" for r in meta])
held=train&(group==11)
source=previous/"audit.py"
proof=read(previous/"result.json")
assert digest(source) in proof["hashes"].values()
opts=proof["summary"]["latest_and_changes"]
wanted={"weighted_ridge","represent","fit_presence"}
tree=ast.parse(source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],
                        type_ignores=[]),str(source),"exec"))
with np.load(data,allow_pickle=False) as packed:
    x=represent(contrast(packed["direct"]),"latest_and_changes")
    y=represent(contrast(packed["reflected"]),"latest_and_changes")
    assert packed["reflected_available"][train].all()
model=fit_presence(x,y,train&~held,opts["alpha"])
mean,scale,offset,beta=model
shape=(-1,3,len(VIEW_ORDER),8,16)
parts=.5*((((x[held]-mean)/scale)*beta).reshape(shape).sum(axis=(3,4))+
          (((y[held]-mean)/scale)*beta).reshape(shape).sum(axis=(3,4))[:,:,[0,2,1,3]])
score=offset+parts.sum(axis=(1,2))
assert np.allclose(score,.5*(predict(model,x[held])+predict(model,y[held])))
record=previous/"rows.jsonl"
assert digest(record)==proof["rows_sha256"]
expected={(r["run"],r["frame"]):r for r in map(json.loads,record.read_text().splitlines())
          if r["cohort"]=="monza_train_oof" and r["mode"]=="latest_and_changes"}
audit_rows=out/"rows.jsonl"
assert digest(audit_rows)==read(out/"result.json")["rows_sha256"]
kind={(r["run"],r["frame"]):r["kind"] for r in map(json.loads,audit_rows.read_text().splitlines())}
rows=[]
for r,s,p in zip((r for r,h in zip(meta,held,strict=True) if h),score,parts,strict=True):
    key=r["run"],r["frame"]
    check=expected[key]
    assert abs(s-check["score"])<1e-9
    rows.append(dict(run=key[0],frame=key[1],near=check["near"],alert=check["alert"],
                     kind=kind.get(key),score=float(s),contributions=p.tolist()))
summary={}
for name,subset in (("straight_false",[r for r in rows if r["kind"]=="onset_beyond_60m"]),
                    ("true_near",[r for r in rows if r["near"]])):
    median=np.median([r["contributions"] for r in subset],axis=0)
    summary[name]=dict(n=len(subset),score_median=float(np.median([r["score"] for r in subset])),
        by_input={f:dict(zip(VIEW_ORDER,values.tolist())) for f,values in
                  zip(("latest","earlier_change","latest_change"),median,strict=True)})
jsonl(out/"contributions.jsonl",rows)
write(out/"contribution-result.json",dict(summary=summary,group=11,seed=None,termination="archive_end",
    revision=proof["revision"],rows_sha256=digest(out/"contributions.jsonl"),
    hashes={str(p):digest(p) for p in (source,data,index,record,audit_rows,Path(__file__))}))
print(summary)
