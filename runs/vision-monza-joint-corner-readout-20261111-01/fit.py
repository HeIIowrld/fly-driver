"""Compare one- and two-lap corner presence heads on the same physical holdout."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from infer_frozen_next_corner import load_readout
from train_vision_teacher_line import digest,jsonl,predict,read,write

out=Path(__file__).parent
root=out.parent
inputs=root/"vision-monza-two-lap-presence-20261111-01"
frozen=root/"vision-frozen-next-corner-20261110-01"
source=frozen/"archive-inference/rows.jsonl"
labels=root/"vision-magione-corner-geometry-front-far-20261111-01/rows.jsonl"
def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines()]
def contrast(raw):
    f=raw.astype(float).reshape(-1,3,4,8,16)
    return (f-f.mean(axis=-1,keepdims=True)).reshape(len(f),-1)
def fit(points,target,weights,alpha):
    mean=np.average(points,axis=0,weights=weights)
    scale=np.sqrt(np.average((points-mean)**2,axis=0,weights=weights)).clip(.04)
    offset=np.average(target,axis=0,weights=weights)
    w=np.sqrt(weights)
    z=(points-mean)/scale*w[:,None]
    gram=z@z.T
    gram.flat[::len(gram)+1]+=alpha
    return mean,scale,offset,z.T@solve(gram,(target-offset)*w[:,None],assume_a="pos")

receipt=read(inputs/"flyvis-result.json")
assert digest(inputs/"rows.jsonl")==receipt["rows_sha256"]
assert digest(inputs/"features.npz")==receipt["features_sha256"]
monza=rows(inputs/"rows.jsonl")
with np.load(inputs/"features.npz",allow_pickle=False) as z:
    x,y=map(contrast,(z["direct"],z["reflected"]))
    mirrored=z["reflected_available"]
train,val=[np.array([r["split"]==split for r in monza]) for split in ("train","validation")]
def category(r):
    if not r["direction"] or not 4<=r["distance"]<=20:
        return 0
    return 1+3*(r["direction"]-1)+int(np.searchsorted([8,12],r["distance"],side="right"))
category_train=np.array([category(r) for r in monza])
swap=np.array([0,4,5,6,1,2,3])
centers=np.array([6.,10.,16.])
def outputs(model,a,b,available):
    score=predict(model,a)
    score[available]=.5*(score[available]+predict(model,b[available])[:,swap])
    score=np.maximum(0.,score)
    sides=score[:,1:].reshape(-1,2,3)
    group=sides.sum(2)
    winner=group.argmax(1)
    values=sides[np.arange(len(a)),winner]
    mass=values.sum(1)
    estimate=np.divide(values@centers,mass,out=np.full(len(a),np.nan),where=mass>0)
    return winner+1,estimate,mass-score[:,0]

def metrics(predicted,estimate,alert,truth_direction,truth_distance):
    near=(truth_direction!=0)
    correct=alert&near&(predicted==truth_direction)
    accurate=correct&(abs(estimate-truth_distance)<=5.)
    return dict(out_of_band=int(sum(alert&~near)),direction_hit=int(sum(correct)),
                direction_distance_hit=int(sum(accurate)),left=int(sum(correct&(truth_direction==1))),
                right=int(sum(correct&(truth_direction==2))),
                long_distance_hit=int(sum(accurate&(truth_distance>=12.))))

v=[r for r,k in zip(monza,val,strict=True) if k]
v_direction=np.array([r["direction"] if category(r) else 0 for r in v])
v_distance=np.array([r["distance"] if r["distance"] is not None else np.inf for r in v])
models={}
validation={}
for name,mask in (("one_lap",train&np.array([r["run"]==monza[0]["run"] for r in monza])),("two_laps",train)):
    assert mirrored[mask].all()
    points=np.vstack((x[mask],y[mask]))
    cats=category_train[mask]
    grouping=[(r["physical_group"],0 if c==0 else (c-1)//3+1)
              for r,c,k in zip(monza,category_train,mask,strict=True) if k]
    groups=Counter(grouping)
    classes=Counter(c for _,c in groups)
    weights=np.tile([1/(groups[g,c]*classes[c]) for g,c in grouping],2)
    weights/=weights.mean()
    target=np.eye(7)[np.r_[cats,swap[cats]]]
    choices=[]
    for alpha in (10.,100.,1000.):
        model=fit(points,target,weights,alpha)
        side,distance,score=outputs(model,x[val],y[val],mirrored[val])
        for threshold in np.r_[np.unique(score),np.nextafter(score.max(),np.inf)]:
            alert=(score>=threshold)&np.isfinite(distance)
            stats=metrics(side,distance,alert,v_direction,v_distance)
            if stats["out_of_band"]<=3:
                rank=(min(stats["left"],stats["right"]),stats["direction_distance_hit"],
                      stats["direction_hit"],-stats["out_of_band"],threshold)
                choices.append((rank,alpha,float(threshold),model,stats))
    _,alpha,cut,model,stats=max(choices,key=lambda item:item[0])
    models[name]=model,cut
    validation[name]=dict(stats,alpha=alpha,threshold=cut)
    np.savez_compressed(out/f"{name}.npz",**dict(zip(("mean","scale","offset","coefficients"),model)),
                        threshold=cut,centers=centers)

base=frozen/"archive-parity"
def external(branch):
    path=base/branch/"features.npz"
    proof=read(path.with_name("flyvis-result.json"))
    assert digest(path)==proof["features_sha256"] and digest(base/"index.jsonl")==proof["rows_sha256"]
    with np.load(path,allow_pickle=False) as z:
        assert z["available"].all()
        return contrast(z["features"])
a,b=map(external,("original","reflected"))
for p in (source,labels):
    assert digest(p)==read(p.with_name("result.json"))["rows_sha256"]
f=rows(source)
order=rows(base/"index.jsonl")
assert [(r["run"],r["frame"]) for r in f]==[(r["run"],r["frame"]) for r in order]
truth={(r["run"],r["frame"]):r for r in rows(labels)}
keyed=[truth[r["run"],r["frame"]] for r in f]
d=np.array([r["label"]["direction"] if r["near"] else 0 for r in keyed])
s=np.array([r["label"]["distance"] if r["near"] else np.inf for r in keyed])
variants={"frozen":(np.array([r["direction"] for r in f]),
                    np.array([r["distance_m"] if r["distance_m"] is not None else np.nan for r in f]),
                    np.array([r["alert"] for r in f]))}
for name,(model,cut) in models.items():
    side,distance,score=outputs(model,a,b,np.ones(len(a),bool))
    variants[name]=side,distance,(score>=cut)&np.isfinite(distance)
summary={name:metrics(*result,d,s) for name,result in variants.items()}
reports=[]
for i,r in enumerate(f):
    predictions={}
    for name,(side,distance,alert) in variants.items():
        predictions[name]=dict(direction=int(side[i]),distance_m=float(distance[i]) if np.isfinite(distance[i]) else None,
                               alert=bool(alert[i]))
    reports.append(dict(run=r["run"],frame=r["frame"],predictions=predictions))
jsonl(out/"predictions.jsonl",reports)
write(out/"result.json",dict(validation=validation,summary=summary,n=len(reports),seed=None,
    revision=receipt["revision"],termination="archive_end",rows_sha256=digest(out/"predictions.jsonl"),
    inputs_sha256={str(p):digest(p) for p in (source,labels,inputs/"flyvis-result.json",Path(__file__),
        base/"original/flyvis-result.json",base/"reflected/flyvis-result.json")}))
print(validation,summary)