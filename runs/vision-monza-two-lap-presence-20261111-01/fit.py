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
frozen=root/"vision-frozen-next-corner-20261110-01"
pins=frozen/"pins.json"
source=frozen/"archive-inference/rows.jsonl"
labels=root/"vision-magione-corner-geometry-front-far-20261111-01/rows.jsonl"
def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines()]
def contrast(raw):
    f=raw.astype(float).reshape(-1,3,4,8,16)
    return (f-f.mean(axis=-1,keepdims=True)).reshape(len(f),-1)
receipt=read(out/"flyvis-result.json")
assert digest(out/"rows.jsonl")==receipt["rows_sha256"]
assert digest(out/"features.npz")==receipt["features_sha256"]
m=rows(out/"rows.jsonl")
with np.load(out/"features.npz",allow_pickle=False) as z:
    raw,mirror=z["direct"],z["reflected"]
    x,y=map(contrast,(raw,mirror))
    has_mirror=z["reflected_available"]
train,val=[np.array([r["split"]==split for r in m]) for split in ("train","validation")]
classes=np.array([r["direction"] if r["direction"] and 4<=r["distance"]<=20 else 0 for r in m])
v_class=classes[val]
swap=np.array([0,2,1])
readout,threshold=load_readout(read(pins))
v_prediction=[]
for i in np.flatnonzero(val):
    d,s,p=readout(raw[i:i+1],mirror[i:i+1] if has_mirror[i] else None)
    v_prediction.append((int(d[0]),float(s[0]),float(p[0])))
v_direction,v_distance,v_presence=np.asarray(v_prediction).T
valid_forecast=(v_direction!=0)&(v_distance<=20)

def validation_stats(alert):
    return dict(false_alerts=int(sum(alert & (v_class==0))),
        direction_hits=int(sum(alert & (v_class!=0)&(v_direction==v_class))),
        by_side=[int(sum(alert & (v_class==d)&(v_direction==d))) for d in (1,2)])
def paired_score(model, a, b, available):
    direct=predict(model,a)
    paired=direct.copy()
    paired[available]=.5*(direct[available]+predict(model,b[available])[:,swap])
    return paired[:,1:].max(1)
def model_input(mask):
    keys=[(r["physical_group"],c) for r,c,k in zip(m,classes,mask,strict=True) if k]
    counts=Counter(keys)
    groups=Counter(c for _,c in counts)
    weights=np.array([1/(groups[c]*counts[g,c]) for g,c in keys])
    weights=np.tile(weights,2)
    weights/=weights.mean()
    points=np.vstack((x[mask],y[mask]))
    category=classes[mask]
    return points,np.eye(3)[np.r_[category,swap[category]]],weights
def fit(points,target,weights,alpha):
    mean=np.average(points,axis=0,weights=weights)
    scale=np.sqrt(np.average((points-mean)**2,axis=0,weights=weights)).clip(.04)
    offset=np.average(target,axis=0,weights=weights)
    w=np.sqrt(weights)
    z=(points-mean)/scale*w[:,None]
    gram=z@z.T
    gram.flat[::len(gram)+1]+=alpha
    return mean,scale,offset,z.T@solve(gram,(target-offset)*w[:,None],assume_a="pos")

models={}
validation={"frozen":validation_stats(valid_forecast & (v_presence>=threshold))}
for name,mask in (("one_lap",train & np.array([r["run"]==m[0]["run"] for r in m])),("two_laps",train)):
    assert has_mirror[mask].all()
    points,target,weights=model_input(mask)
    choices=[]
    for alpha in (10.,100.,1000.):
        model=fit(points,target,weights,alpha)
        score=paired_score(model,x[val],y[val],has_mirror[val])
        for cut in np.r_[np.unique(score),np.nextafter(score.max(),np.inf)]:
            stats=validation_stats(valid_forecast & (score>=cut))
            if stats["false_alerts"]<=2:
                side=stats["by_side"]
                choices.append(((min(side),sum(side),-stats["false_alerts"],cut),
                                alpha,float(cut),model,stats))
    _,alpha,cut,model,stats=max(choices,key=lambda item:item[0])
    models[name]=model,cut
    validation[name]=dict(stats,alpha=alpha,threshold=cut)
    np.savez_compressed(out/f"{name}.npz",**dict(zip(("mean","scale","offset","coefficients"),model)),
                        threshold=cut)

base=frozen/"archive-parity"
def external(branch):
    p=base/branch/"features.npz"
    proof=read(p.with_name("flyvis-result.json"))
    assert digest(p)==proof["features_sha256"] and digest(base/"index.jsonl")==proof["rows_sha256"]
    with np.load(p,allow_pickle=False) as z:
        assert z["available"].all()
        return contrast(z["features"])
a,b=map(external,("original","reflected"))
assert digest(source)==read(source.with_name("result.json"))["rows_sha256"]
assert digest(labels)==read(labels.with_name("result.json"))["rows_sha256"]
forecasts=rows(source)
order=rows(base/"index.jsonl")
assert [(r["run"],r["frame"]) for r in forecasts]==[(r["run"],r["frame"]) for r in order]
truth={(r["run"],r["frame"]):r for r in rows(labels)}
scores={name:(paired_score(model,a,b,np.ones(len(a),bool)),cut) for name,(model,cut) in models.items()}
reports=[]
for i,r in enumerate(forecasts):
    label=truth[r["run"],r["frame"]]
    direction,distance=r["direction"],r["distance_m"]
    alerts={"frozen":r["alert"]}
    alerts.update({name:bool(direction and distance<=20 and score[i]>=cut) for name,(score,cut) in scores.items()})
    reports.append(dict(run=r["run"],frame=r["frame"],near=label["near"],label=label["label"],
                        direction=direction,distance_m=distance,alerts=alerts))
summary={}
for name in ("frozen",*models):
    c=Counter()
    for r in reports:
        if not r["alerts"][name]:
            continue
        if not r["near"]:
            c["out_of_band"]+=1
            if r["label"] and (not r["label"]["direction"] or r["label"]["distance"]>30):
                c["beyond_30_or_no_labelled_corner"]+=1
        elif r["direction"]==r["label"]["direction"]:
            c[f"direction_{r['direction']}"]+=1
            c["distance_hit"]+=abs(r["distance_m"]-r["label"]["distance"])<=5
    summary[name]=dict(c)
jsonl(out/"predictions.jsonl",reports)
write(out/"result.json",dict(validation=validation,summary=summary,n=len(reports),seed=None,
    revision=receipt["revision"],termination="archive_end",rows_sha256=digest(out/"predictions.jsonl"),
    inputs_sha256={str(p):digest(p) for p in (source,labels,pins,out/"flyvis-result.json",Path(__file__),
        base/"original/flyvis-result.json",base/"reflected/flyvis-result.json")}))
print(validation,summary)
