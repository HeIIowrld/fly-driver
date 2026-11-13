from collections import Counter
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve
from train_next_corner import macro
from train_vision_teacher_line import digest,fit,jsonl,predict,read,write

out=Path(__file__).parent
base=Path("runs/vision-monza-midpoint-raw-head-20261110-01")
old=Path("runs/vision-midpoint-aligned-head-20261110-01")
magione=Path("runs/vision-magione-edge-midpoint-inventory-20261110-01")
source=read(base/"result.json")
flyvis=read(base/"flyvis-result.json")
external=read(magione/"flyvis-result.json")
def load(path,expected):
    assert digest(path/"rows.jsonl")==expected["rows_sha256"]
    assert digest(path/"features.npz")==expected["features_sha256"]
    rows=[json.loads(line) for line in (path/"rows.jsonl").read_text().splitlines()]
    with np.load(path/"features.npz",allow_pickle=False) as z:
        x=z["features"][z["available"]].astype(float)
        rows=[r for r,k in zip(rows,z["available"],strict=True) if k]
    return rows,x
monza,x=load(base,flyvis)
ext,xr=load(magione,external)
ext=[dict(r,split="external") for r in ext]
assert {r["physical_group"] for r in monza if r["split"]=="train"}.isdisjoint(
    {r["physical_group"] for r in monza if r["split"]=="validation"})
candidate_groups=source["near_groups"]
held=[]
validation_groups=source["validation_groups"]
counts={}
reflection=read(out/"flyvis-result.json")
assert reflection["rows_sha256"]==flyvis["rows_sha256"]
assert digest(out/"features.npz")==reflection["features_sha256"]
with np.load(base/"features.npz",allow_pickle=False) as z:
    original_available=z["available"].copy()
source_rows=[json.loads(line) for line in (base/"rows.jsonl").read_text().splitlines()]
with np.load(out/"features.npz",allow_pickle=False) as z:
    keep=z["available"]
    assert np.array_equal(keep,original_available & np.array([r["split"]=="train" for r in source_rows]))
    reflected=z["features"][keep].astype(float)
augmented=[dict(r,split="reflected",direction=(0,2,1)[r["direction"]])
           for r,k in zip(source_rows,keep,strict=True) if k]
data=monza+ext+augmented
raw_features=np.vstack((x,xr,reflected))
direction=np.array([r["direction"] for r in data])
distance=np.array([r["distance"] if r["distance"] is not None else np.inf for r in data])
split=np.array([r["split"] for r in data])
target=np.where((direction!=0)&(distance<=30),direction,0)
train,val=split=="train",split=="validation"
assert set(target[train])==set(target[val])=={0,1,2}

def weighted_head(x,labels,ridge):
    w=len(labels)/(3*np.bincount(labels,minlength=3)[labels])
    mu=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mu)**2,axis=0,weights=w)).clip(.04)
    offset=np.average(np.eye(3)[labels],axis=0,weights=w)
    root=np.sqrt(w)[:,None]
    a=(x-mu)/scale*root
    gram=a@a.T
    gram.flat[::len(gram)+1]+=ridge
    return mu,scale,offset,a.T@solve(gram,(np.eye(3)[labels]-offset)*root,assume_a="pos")
reports={}
lattice=raw_features.reshape(-1,3,4,8,16)
centered=lattice-lattice.mean(axis=-1,keepdims=True)
variants={"spatial_contrast":centered,"reflected_training":centered}
original_train=train.copy()
for variant,array in variants.items():
    train=original_train | (split=="reflected") if variant=="reflected_training" else original_train
    features=array.reshape(len(data),-1)
    heads=[]
    for alpha in (10.,100.,1000.):
        cls=weighted_head(features[train],target[train],alpha)
        scores=predict(cls,features[val])
        for bias in (0.,.1,.2,.3):
            p=(scores+[bias,0.,0.]).argmax(1)
            heads.append(((macro(target[val],p),-np.mean(p[target[val]==0]!=0)),alpha,bias,cls))
    _,alpha,bias,cls=max(heads,key=lambda a:a[0])
    pos=train&(target!=0)
    near_val=val&(target!=0)
    regressors=[]
    for ridge in (10.,100.,1000.):
        reg=fit(features[pos],distance[pos],ridge)
        err=np.abs(predict(reg,features[near_val]).clip(0.,30.)-distance[near_val]).mean()
        regressors.append((err,ridge,reg))
    _,ridge,reg=min(regressors,key=lambda r:r[0])
    pred=(predict(cls,features)+[bias,0.,0.]).argmax(1)
    dist=predict(reg,features).clip(0.,30.)
    metrics={}
    for name in ("train","validation","external"):
        group=split==name
        near=group&(direction!=0)&(distance>=4)&(distance<=20)
        hit=near&(pred==direction)
        alert=(pred!=0)&(dist<=20)
        metrics[name]=dict(n=int(group.sum()),class_counts=np.bincount(target[group],minlength=3).tolist(),
            near_n=int(near.sum()),direction_hits=int(hit.sum()),timely_direction_hits=int((hit&alert).sum()),
            hits_within_5m=int((hit&alert&(abs(dist-distance)<=5)).sum()),
            far_or_none_alerts=int((group&(distance>30)&alert).sum()),
            conditional_near_distance_mae_m=float(np.mean(abs(dist[near]-distance[near]))) if near.any() else None)
    weights=out/f"{variant}-weights.npz"
    keys=("mean","scale","offset","coefficients")
    np.savez_compressed(weights,**{f"cls_{k}":v for k,v in zip(keys,cls,strict=True)},
        **{f"dist_{k}":v for k,v in zip(keys,reg,strict=True)},bias=bias)
    jsonl(out/f"{variant}-predictions.jsonl",(dict(r,predicted=int(p),predicted_distance_m=float(d) if p else None)
        for r,p,d in zip(data,pred,dist,strict=True)))
    reports[variant]=dict(metrics=metrics,
        train_counts_by_group=candidate_groups,held_groups=sorted(held),validation_groups=validation_groups,
        selection=dict(alpha=alpha,bias=bias,ridge=ridge),source_drops=dict(counts),
        weights_sha256=digest(weights),predictions_sha256=digest(out/f"{variant}-predictions.jsonl"),
        feature_dim=int(features.shape[1]))

prior=read(base/"readout-result.json")
control=read(Path("runs/vision-midpoint-retinal-representation-20261110-01/result.json"))
assert reports["spatial_contrast"]["metrics"]==control["reports"]["spatial_contrast"]["metrics"]
write(out/"result.json",dict(revision=prior["revision"],seed=None,reports=reports,
    inputs_sha256={str(p):digest(p) for p in (Path(__file__),base/"readout-result.json",base/"flyvis-result.json",
                 magione/"flyvis-result.json",base/"fit.py",out/"flyvis-result.json")},
    scope="reflected Monza train inputs only; no neuron permutation; Magione diagnostic only",
    termination="completed fixed sample representation comparison"))
for name,r in reports.items():
    print(name,r["metrics"]["validation"],r["metrics"]["external"])
