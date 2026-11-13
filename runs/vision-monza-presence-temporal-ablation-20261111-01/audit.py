"""Compare temporal pooling for near-corner presence in the same physical folds."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from infer_frozen_next_corner import contrast, load_features
from train_vision_teacher_line import digest, jsonl, predict, read, write

out=Path(__file__).parent
root=out.parent
src=root/"vision-monza-two-lap-presence-20261111-01"
mono=root/"vision-monza-corner-event-readout-20261111-01/rows.jsonl"
mag=root/"vision-magione-left-readout-stages-20261111-01/rows.jsonl"
frozen=root/"vision-frozen-next-corner-20261110-01/archive-parity"
p=src/"features.npz"
proof=read(src/"flyvis-result.json")
assert digest(p)==proof["features_sha256"]

def load(path):
    assert digest(path)==read(path.with_name("result.json"))["rows_sha256"]
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]

rows,external=load(mono),load(mag)
index=frozen/"index.jsonl"
keys=lambda group:[(r["run"],r["frame"]) for r in group]
index_rows=[json.loads(line) for line in index.read_text(encoding="utf8").splitlines()]
assert keys(external)==keys(index_rows)
monza_index=[json.loads(line) for line in (src/"rows.jsonl").read_text(encoding="utf8").splitlines()]
assert digest(src/"rows.jsonl")==proof["rows_sha256"] and keys(rows)==keys(monza_index)
with np.load(p,allow_pickle=False) as z:
    x,y=map(contrast,(z["direct"],z["reflected"]))
    mirrored=z["reflected_available"]

group=np.array([r["physical_group"] for r in rows])
train=np.array([r["split"]=="train" for r in rows])
val=~train
truth=np.array([r["actual"] if r["near"] else 0 for r in rows])
assert set(group[train]).isdisjoint(group[val]) and mirrored[train].all()

def weighted_ridge(points,target,w,alpha):
    mean=np.average(points,axis=0,weights=w)
    scale=np.sqrt(np.average((points-mean)**2,axis=0,weights=w)).clip(.04)
    offset=np.average(target,axis=0,weights=w)
    root_w=np.sqrt(w)
    z=((points-mean)/scale)*root_w[:,None]
    gram=z@z.T
    gram.flat[::len(gram)+1]+=alpha
    beta=z.T@solve(gram,(target-offset)*root_w,assume_a="pos")
    return mean,scale,offset,beta

def stats(pred,rows):
    target=np.array([r["actual"] if r["near"] else 0 for r in rows])
    return dict(left=int(np.sum((pred==1)&(target==1))),right=int(np.sum((pred==2)&(target==2))),
        left_n=int(np.sum(target==1)),right_n=int(np.sum(target==2)),
        outside=int(np.sum((pred!=0)&(target==0))),
        far=int(sum(s and not r["near"] and (not r["actual"] or r.get("onset_m",r.get("distance")) is None
                    or r.get("onset_m",r.get("distance"))>30.) for s,r in zip(pred,rows,strict=True))))

def baseline(rows):
    return np.array([r["raw"] if r["alert"] else 0 for r in rows])

from scipy.stats import rankdata

def represent(features, mode):
    history=features.reshape(len(features),3,-1)
    if mode=="latest":
        return history[:,-1]
    if mode=="changes":
        return np.diff(history,axis=1).reshape(len(features),-1)
    if mode=="latest_and_changes":
        return np.concatenate((history[:,-1:],np.diff(history,axis=1)),axis=1).reshape(len(features),-1)
    return features

def fit_presence(a,b,mask,alpha):
    labels=truth[mask]
    pairs=list(zip(group[mask],labels,strict=True))
    counts=Counter(pairs)
    sides=Counter(side for _,side in counts)
    weight=np.tile([1/(sides[side]*counts[g,side]) for g,side in pairs],2)
    weight/=weight.mean()
    return weighted_ridge(np.vstack((a[mask],b[mask])),np.tile(labels!=0,2).astype(float),weight,alpha)

def presence_scores(model,a,b,available):
    scores=predict(model,a)
    scores[available]=.5*(scores[available]+predict(model,b[available]))
    return scores

def presence_stats(scores,cut,cohort):
    labels=np.array([r["actual"] if r["near"] else 0 for r in cohort])
    eligible=np.array([r["in_range"] for r in cohort])
    alerted=(scores>=cut)&eligible
    result={}
    for side in (1,2):
        matching=(labels==side)|(labels==0)
        local_ranks=rankdata(scores[matching])
        positive=labels[matching]==side
        n=int(positive.sum())
        m=int((~positive).sum())
        result[str(side)]=dict(n=n,hit=int(sum(alerted&(labels==side))),
            auc=float((sum(local_ranks[positive])-n*(n+1)/2)/(n*m)))
    return dict(by_side=result,outside=int(sum(alerted&(labels==0))))

train_rows=[r for r,keep in zip(rows,train,strict=True) if keep]
valid_rows=[r for r,keep in zip(rows,val,strict=True) if keep]
budget=stats(baseline(train_rows),train_rows)["outside"]
files=[frozen/branch/"features.npz" for branch in ("original","reflected")]
loaded=[load_features(f,digest(index),len(external)) for f in files]
assert all(available.all() for _,available,_ in loaded)
external_a,external_b=[contrast(v[0]) for v in loaded]
report={}
outputs=[]
for mode in ("latest","full_history","changes","latest_and_changes"):
    a,b=represent(x,mode),represent(y,mode)
    choices=[]
    for alpha in (100.,1000.,10000.):
        scores=np.empty(len(rows))
        for held in sorted(set(group[train])):
            fold=train&(group==held)
            model=fit_presence(a,b,train&~fold,alpha)
            scores[fold]=presence_scores(model,a[fold],b[fold],mirrored[fold])
        scores=scores[train]
        labels=truth[train]
        eligible=np.array([r["in_range"] for r in train_rows])
        for cut in np.r_[np.unique(scores),np.nextafter(scores.max(),np.inf)]:
            alert=(scores>=cut)&eligible
            outside=int(sum(alert&(labels==0)))
            if outside<=budget:
                rates=[np.mean(alert[labels==side]) for side in (1,2)]
                choices.append(((min(rates),sum(rates),-outside,cut),alpha,float(cut),scores))
    _,alpha,cut,cv=max(choices,key=lambda choice:choice[0])
    model=fit_presence(a,b,train,alpha)
    validation=presence_scores(model,a[val],b[val],mirrored[val])
    transfer=presence_scores(model,represent(external_a,mode),represent(external_b,mode),
                             np.ones(len(external),dtype=bool))
    report[mode]=dict(alpha=alpha,threshold=cut,train_group_cv=presence_stats(cv,cut,train_rows),
                      monza_validation=presence_stats(validation,cut,valid_rows),
                      magione_development=presence_stats(transfer,cut,external))
    report[mode]["oof_by_physical_group"]={}
    for held in sorted(set(group[train])):
        mask=group[train]==held
        subset=[r for r,keep in zip(train_rows,mask,strict=True) if keep]
        values=cv[mask]
        ranges={}
        for label in (0,1,2):
            score=[float(v) for r,v in zip(subset,values,strict=True)
                   if (r["actual"] if r["near"] else 0)==label]
            if score:
                ranges[str(label)]={"n":len(score),"min":min(score),"max":max(score),
                                    "alerts":sum(v>=cut for v in score)}
        report[mode]["oof_by_physical_group"][str(int(held))]=ranges
    for name,group_rows,values in (("monza_train_oof",train_rows,cv),("monza_validation",valid_rows,validation),
                                   ("magione_development",external,transfer)):
        outputs.extend(dict(mode=mode,cohort=name,run=r["run"],frame=r["frame"],score=float(value),actual=r["actual"],near=r["near"],
                            alert=bool(value>=cut and r["in_range"]))
                       for r,value in zip(group_rows,values,strict=True))
jsonl(out/"rows.jsonl",outputs)
write(out/"result.json",dict(summary=report,train_outside_budget=budget,seed=None,termination="archive_end",
    revision=proof["revision"],rows_sha256=digest(out/"rows.jsonl"),
    hashes={str(f):digest(f) for f in (mono,mag,p,index,src/"rows.jsonl",Path(__file__),*files)}))
for mode,scores in report.items():
    print(mode,scores)
