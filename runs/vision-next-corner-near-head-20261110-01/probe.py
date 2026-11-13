from collections import Counter
import json
import subprocess
from pathlib import Path
import numpy as np
from train_next_corner import CLASSES, macro
from train_vision_teacher_line import digest, fit, jsonl, predict, read, write

out=Path(__file__).parent
source=Path("runs/vision-next-corner-20261110-03")
receipt=read(source/"result.json")
assert digest(source/"rows.jsonl")==receipt["rows_sha256"]
assert digest(source/"flyvis-features.npz")==receipt["flyvis_features_sha256"]
raw=[json.loads(line) for line in (source/"rows.jsonl").read_text().splitlines()]
excluded={(r["run"],i) for r in raw if r["split"]=="excluded" for i in range(r["frame"]-2,r["frame"]+1)}
clean=np.array([r["split"] not in ("train","validation") or not any(
    (r["run"],i) in excluded for i in range(r["frame"]-2,r["frame"]+1)) for r in raw])
assert int((~clean).sum())==receipt["excluded_history_rows_removed"]
with np.load(source/"flyvis-features.npz",allow_pickle=False) as z:
    keep=clean & z["available"]
    flyvis=z["features"][keep].astype(float)
pixels=np.load(source/"image-features.npy")[keep].astype(float)
rows=[r for r,k in zip(raw,keep) if k]
split=np.array([r["split"] for r in rows])
train,validation=split=="train",split=="validation"
distance=np.array([r["distance"] if r["distance"] is not None else np.inf for r in rows])
direction=np.array([r["direction"] for r in rows])
near=(direction!=0)&(distance<=30.)
labels=np.where(near,direction,0)
assert set(labels[train])==set(labels[validation])=={0,1,2}
assert all(rows[i]["run"]=="monza-user-started-teacher-01" for i in np.flatnonzero(train|validation))
assert not {(r["run"],r["event"]) for r,m in zip(rows,train&near) if m} & {
    (r["run"],r["event"]) for r,m in zip(rows,validation&near) if m}
models={}
for name,x in (("flyvis",flyvis),("pixels",pixels),("pixels_plus_flyvis",np.column_stack((pixels,flyvis)))):
    alternatives=[]
    for alpha in (10.,100.,1000.):
        cls=fit(x[train],np.eye(3)[labels[train]],alpha)
        scores=predict(cls,x[validation])
        for bias in (-.30,-.15,0.):
            p=(scores+[bias,0.,0.]).argmax(1)
            score=(macro(labels[validation],p),-float(np.mean(p[labels[validation]==0]!=0)))
            alternatives.append((score,alpha,bias,cls))
    score,alpha,bias,cls=max(alternatives,key=lambda a:a[0])
    fit_dist=train&near
    val_dist=validation&near
    alternatives=[]
    for ridge in (10.,100.,1000.):
        reg=fit(x[fit_dist],distance[fit_dist],ridge)
        error=np.abs(predict(reg,x[val_dist]).clip(0,30)-distance[val_dist]).mean()
        alternatives.append((float(error),ridge,reg))
    error,ridge,reg=min(alternatives,key=lambda a:a[0])
    category=(predict(cls,x)+[bias,0.,0.]).argmax(1)
    predicted_distance=predict(reg,x).clip(0,30.)
    metrics={}
    for role in ("train","validation","heldout","test_other","external"):
        group=split==role
        hits=group&near&(category==direction)
        close=group&near&(distance<=10)
        metrics[role]=dict(n=int(group.sum()),near_n=int((group&near).sum()),
            correct_direction_n=int(hits.sum()),far_or_none_alert_n=int((group&~near&(category!=0)).sum()),
            far_or_none_n=int((group&~near).sum()),close_n=int(close.sum()),
            close_correct_n=int((hits&close).sum()),
            close_correct_and_within_5m_n=int((hits&close&(abs(predicted_distance-distance)<=5)).sum()),
            distance_mae_on_direction_hits_m=float(np.abs(predicted_distance[hits]-distance[hits]).mean()) if hits.any() else None)
    keys=("mean","scale","offset","coefficients")
    weights=out/f"{name}-weights.npz"
    np.savez_compressed(weights,**{f"cls_{k}":v for k,v in zip(keys,cls,strict=True)},
        **{f"dist_{k}":v for k,v in zip(keys,reg,strict=True)},bias=bias)
    predictions=out/f"{name}-predictions.jsonl"
    jsonl(predictions,(dict(r,predicted=CLASSES[c],predicted_distance_m=float(d) if c else None)
                       for r,c,d in zip(rows,category,predicted_distance,strict=True)))
    models[name]=dict(direction_alpha=alpha,bias=bias,distance_alpha=ridge,
        validation_distance_mae=error,metrics=metrics,weights_sha256=digest(weights),
        predictions_sha256=digest(predictions))
write(out/"result.json",dict(seed=None,horizon_m=30.,models=models,
    revision=subprocess.check_output(["git","-c","safe.directory=*","rev-parse","HEAD"],text=True).strip(),
    source_sha256={str(p):digest(p) for p in (source/"result.json",source/"rows.jsonl",source/"flyvis-features.npz",
                   source/"image-features.npy",Path(__file__))},
    train_near_groups=dict(Counter(str((r["event"],r["direction"])) for r,m in zip(rows,train&near) if m)),
    val_near_groups=dict(Counter(str((r["event"],r["direction"])) for r,m in zip(rows,validation&near) if m)),
    scope="Monza train/validation only; external evaluation frozen",termination="completed"))
for name,report in models.items():
    print(name,report["metrics"]["validation"],report["metrics"]["external"])
