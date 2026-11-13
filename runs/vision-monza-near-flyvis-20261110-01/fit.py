from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from train_next_corner import CLASSES,macro
from train_vision_teacher_line import digest,fit,jsonl,predict,read,write

out=Path(__file__).parent
original=Path("runs/vision-next-corner-20261110-03")
original_fit=Path("runs/vision-next-corner-near-head-20261110-01")
inventory=Path("runs/vision-monza-near-archive-inventory-20261110-02")
existing=read(original/"result.json")
new=read(out/"result.json")
def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
assert digest(original/"rows.jsonl")==existing["rows_sha256"]
assert digest(original/"flyvis-features.npz")==existing["flyvis_features_sha256"]
assert digest(out/"features.npz")==new["features_sha256"]
assert digest(inventory/"rows.jsonl")==new["rows_sha256"]
rows=read_rows(original/"rows.jsonl")
additional=read_rows(inventory/"rows.jsonl")
# Preserve the exact previously admitted Monza and external rows.
old_predictions=original_fit/"flyvis-predictions.jsonl"
assert digest(old_predictions)==read(original_fit/"result.json")["models"]["flyvis"]["predictions_sha256"]
admitted=read_rows(old_predictions)
ids={(r["run"],r["frame"]) for r in admitted}
old_index={(r["run"],r["frame"]):i for i,r in enumerate(rows)}
assert len(ids)==len(admitted)
with np.load(original/"flyvis-features.npz",allow_pickle=False) as z:
    x=z["features"][[old_index[r["run"],r["frame"]] for r in admitted]].astype(float)
with np.load(out/"features.npz",allow_pickle=False) as z:
    xa=z["features"][z["available"]].astype(float)
    additional=[r for r,k in zip(additional,z["available"],strict=True) if k]
meta=read(original/"dataset.json")["sources"]["monza-user-started-teacher-01"]
stations=np.array(meta["corner_map_m"])
validation_groups={meta["event_groups"][i] for i in meta["validation_events"]}
assert all(meta["event_groups"][r["event"]] not in validation_groups for r in admitted if r["split"]=="train")
for r in additional:
    discrepancy=np.abs(stations-r["map_onset_m"])
    matches=np.flatnonzero(discrepancy<=8.)
    assert len(matches)==1
    event=matches[0]
    group=meta["event_groups"][event]
    assert group not in {meta["event_groups"][i] for i in meta["heldout_events"]}
    r["split"]="archive_validation" if group in validation_groups else "archive_train"
    r["physical_group"]=group
rows=admitted+additional
x=np.vstack((x,xa))
role=np.array([r["split"] for r in rows])
d=np.array([r["distance"] if r["distance"] is not None else np.inf for r in rows])
direction=np.array([r["direction"] for r in rows])
near=(direction!=0)&(d<=30.)
labels=np.where(near,direction,0)
validation=role=="validation"
assert set(labels[validation])=={0,1,2}

def balanced_fit(x,y,alpha):
    counts=np.bincount(y,minlength=3)
    w=len(y)/(3*counts[y])
    mu=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mu)**2,axis=0,weights=w)).clip(.04)
    root=np.sqrt(w)[:,None]
    z=(x-mu)/scale*root
    target=np.eye(3)[y]
    offset=np.average(target,axis=0,weights=w)
    gram=z@z.T
    gram.flat[::len(gram)+1]+=alpha
    return mu,scale,offset,z.T@solve(gram,(target-offset)*root,assume_a="pos")

reports={}
for name,train in (("base",role=="train"),("augmented",(role=="train")|(role=="archive_train"))):
    scores=[]
    for alpha in (10.,100.,1000.):
        cls=balanced_fit(x[train],labels[train],alpha)
        val=predict(cls,x[validation])
        for bias in (0.,.10,.20,.30):
            p=(val+[bias,0.,0.]).argmax(1)
            key=(macro(labels[validation],p),-float(np.mean(p[labels[validation]==0]!=0)))
            scores.append((key,alpha,bias,cls))
    _,alpha,bias,cls=max(scores,key=lambda a:a[0])
    choices=[]
    for ridge in (10.,100.,1000.):
        reg=fit(x[train&near],d[train&near],ridge)
        err=np.abs(predict(reg,x[validation&near]).clip(0.,30.)-d[validation&near]).mean()
        choices.append((float(err),ridge,reg))
    _,ridge,reg=min(choices,key=lambda r:r[0])
    category=(predict(cls,x)+[bias,0.,0.]).argmax(1)
    distance=predict(reg,x).clip(0.,30.)
    metrics={}
    for split in ("validation","archive_validation","heldout","test_other","external"):
        group=role==split
        hit=group&near&(category==direction)
        close=group&near&(d<=10.)
        metrics[split]=dict(n=int(group.sum()),near_n=int((group&near).sum()),hits=int(hit.sum()),
            false_alerts=int((group&~near&(category!=0)).sum()),far_n=int((group&~near).sum()),
            within_5m_and_correct=int((hit&(abs(distance-d)<=5)).sum()),
            within_10m_truth_correct_and_within_5m=int((hit&close&(abs(distance-d)<=5)).sum()),
            conditional_near_distance_mae=float(np.abs(distance[group&near]-d[group&near]).mean()))
    weight=out/f"{name}-weights.npz"
    keys=("mean","scale","offset","coefficients")
    np.savez_compressed(weight,**{f"cls_{k}":v for k,v in zip(keys,cls,strict=True)},
        **{f"dist_{k}":v for k,v in zip(keys,reg,strict=True)},bias=bias)
    pred=out/f"{name}-predictions.jsonl"
    jsonl(pred,(dict(r,predicted=CLASSES[c],predicted_distance_m=float(p) if c else None)
                for r,c,p in zip(rows,category,distance,strict=True)))
    reports[name]=dict(train_counts=np.bincount(labels[train],minlength=3).tolist(),
        metrics=metrics,alpha=alpha,bias=bias,ridge=ridge,weights_sha256=digest(weight),
        predictions_sha256=digest(pred))
write(out/"readout-result.json",dict(revision=new["revision"],seed=None,models=reports,
    added_counts=dict(Counter(r["split"] for r in additional)),
    sources_sha256={str(p):digest(p) for p in (out/"result.json",original/"result.json",original/"dataset.json",
                                             old_predictions,Path(__file__))},
    scope="balanced Monza ridge heads; repeated physical groups share split; Magione evaluation only",
    termination="completed fixed readout comparison"))
for name,r in reports.items():
    print(name,r["train_counts"],r["metrics"]["validation"],r["metrics"]["external"])
