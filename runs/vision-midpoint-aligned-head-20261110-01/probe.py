from collections import Counter
from pathlib import Path
import json

import numpy as np
from scipy.linalg import solve
from scipy.spatial import cKDTree
from train_next_corner import BASE, MONZA, CURVATURE_THRESHOLD, assign, macro
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at,digest,fit,jsonl,predict,read,write

out=Path(__file__).parent
original=Path("runs/vision-next-corner-20261110-03")
magione=Path("runs/vision-magione-edge-midpoint-inventory-20261110-01")
prior=Path("runs/vision-corner-label-geometry-20261110-01")
metadata=read(original/"dataset.json")
receipt=read(original/"result.json")
assert digest(original/"rows.jsonl")==receipt["rows_sha256"]
assert digest(original/"flyvis-features.npz")==receipt["flyvis_features_sha256"]
external=read(magione/"flyvis-result.json")
assert digest(magione/"features.npz")==external["features_sha256"]
assert digest(magione/"rows.jsonl")==external["rows_sha256"]
def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines()]

teacher.CORNER=CURVATURE_THRESHOLD
code=(prior/"probe.py").read_text(encoding="utf-8-sig")
exec(compile(code[code.index("def periodic_line("):code.index("def near(")],str(prior/"probe.py"),"exec"))
mapfile=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
assert digest(mapfile)==metadata["monza_map_sha256"]
road=np.genfromtxt(mapfile,names=True,delimiter=",")
mid=np.c_[(road["left_border_x"]+road["right_border_x"])/2,(road["left_border_y"]+road["right_border_y"])/2]
_,grid,ref_tree,length,holdouts=teacher.road_model(mapfile)
run,sub=MONZA[0]
native,arc,_,provenance=teacher.load_teacher(BASE/run,BASE/run/sub)
assert provenance["tcp_sha256"]==metadata["sources"][run]["provenance"]["tcp_sha256"]
ml,ms,mg,mt=periodic_line(mid)
_,ri=mt.query(native[:,2:4])
if np.median((np.diff(mg[ri])+ml/2)%ml-ml/2)<0:
    ml,ms,mg,mt=periodic_line(mid[::-1])
# Protect the original geographically held-out zones and join nearby bend pieces.
centers=grid[ref_tree.query(mt.data)[1]]
groups=np.cumsum([0]+[int(b[0]-a[1]>20) for a,b in zip(ms,ms[1:])])
held=set()
for i,(start,end,_) in enumerate(ms):
    sample=np.arange(start,end+1)%len(centers)
    if np.any(np.abs((centers[sample,None]-holdouts+length/2)%length-length/2)<=90):
        held.add(int(groups[i]))
all_rows=rows(original/"rows.jsonl")
excluded={(r["run"],i) for r in all_rows if r["split"]=="excluded" for i in range(r["frame"]-2,r["frame"]+1)}
with np.load(original/"flyvis-features.npz",allow_pickle=False) as z:
    kept=[]
    counts=Counter()
    for i,r in enumerate(all_rows):
        if r["run"]!=run or r["split"] not in ("train","validation") or not z["available"][i]:
            continue
        if any((run,k) in excluded for k in range(r["frame"]-2,r["frame"]+1)):
            continue
        t=r["time"]
        if not native[0,0]<=t<native[-1,0]:
            continue
        pos=at(native[:,0],native[:,2:4],t)[0]
        residual,k=mt.query(pos)
        station=float(mg[k])
        label=assign(station,ms,ml,True)
        if residual>7 or label is None:
            counts["invalid_or_inside"]+=1
            continue
        event,direction,distance=label
        group=int(groups[event])
        samples=(np.arange(station-15.,station+min(distance,30.)+1).astype(int))%len(centers)
        if group in held or np.any(np.abs((centers[samples,None]-holdouts+length/2)%length-length/2)<=90):
            counts["heldout_geometry"]+=1
            continue
        kept.append((i,dict(r,event=event,physical_group=group,direction=direction,distance=float(distance) if direction else None)))
    x=z["features"][[i for i,_ in kept]].astype(float)
monza=[r for _,r in kept]
candidate_groups={}
for sign in (1,2):
    c=Counter(r["physical_group"] for r in monza if r["direction"]==sign and r["distance"]<=30)
    candidate_groups[sign]=dict(c)
    assert len(c)>=2, (sign,c)
validation_groups=[]
for sign in (1,2):
    validation_groups.append(max(g for g in candidate_groups[sign] if g not in validation_groups))
for row in monza:
    row["split"]="validation" if row["physical_group"] in validation_groups else "train"
with np.load(magione/"features.npz",allow_pickle=False) as z:
    xr=z["features"][z["available"]].astype(float)
    ext=[dict(r,split="external") for r,k in zip(rows(magione/"rows.jsonl"),z["available"],strict=True) if k]
data=monza+ext
features=np.vstack((x,xr))
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
weights=out/"weights.npz"
keys=("mean","scale","offset","coefficients")
np.savez_compressed(weights,**{f"cls_{k}":v for k,v in zip(keys,cls,strict=True)},
    **{f"dist_{k}":v for k,v in zip(keys,reg,strict=True)},bias=bias)
jsonl(out/"predictions.jsonl",(dict(r,predicted=int(p),predicted_distance_m=float(d) if p else None)
    for r,p,d in zip(data,pred,dist,strict=True)))
write(out/"result.json",dict(revision=external["revision"],seed=None,metrics=metrics,
    train_counts_by_group=candidate_groups,held_groups=sorted(held),validation_groups=validation_groups,
    selection=dict(alpha=alpha,bias=bias,ridge=ridge),source_drops=dict(counts),
    inputs_sha256={str(p):digest(p) for p in (Path(__file__),mapfile,original/"result.json",magione/"flyvis-result.json",prior/"probe.py")},
    weights_sha256=digest(weights),predictions_sha256=digest(out/"predictions.jsonl"),
    scope="midpoint-aligned labels on original Monza camera development only; Magione untouched by fit",
    termination="completed"))
print(candidate_groups,validation_groups,metrics,flush=True)
