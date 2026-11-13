"""Evaluate the joint distance replacement on the original Monza validation groups."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from infer_frozen_next_corner import forecast_values,load_readout
from train_vision_teacher_line import digest,jsonl,predict,read,write

out=Path(__file__).parent
root=out.parent
data=root/"vision-monza-two-lap-presence-20261111-01"
weights=root/"vision-monza-joint-corner-readout-20261111-01/two_laps.npz"
pins=root/"vision-frozen-next-corner-20261110-01/pins.json"
receipt=read(data/"flyvis-result.json")
labels=data/"rows.jsonl"
features=data/"features.npz"
assert digest(labels)==receipt["rows_sha256"] and digest(features)==receipt["features_sha256"]
fit_result=weights.with_name("result.json")
label=[json.loads(s) for s in labels.read_text(encoding="utf8").splitlines()]
frozen,frozen_cut=load_readout(read(pins))
with np.load(weights,allow_pickle=False) as model:
    joint=tuple(model[k] for k in ("mean","scale","offset","coefficients"))
    cut=float(model["threshold"])
    centers=model["centers"]

def contrast(data):
    x=data.astype(float).reshape(-1,3,4,8,16)
    return (x-x.mean(axis=-1,keepdims=True)).reshape(len(x),-1)

with np.load(features,allow_pickle=False) as z:
    direct,mirror,available=z["direct"],z["reflected"],z["reflected_available"]
    x,y=map(contrast,(direct,mirror))
rows=[]
for i,r in enumerate(label):
    d,s,p=frozen(direct[i:i+1],mirror[i:i+1] if available[i] else None)
    base=forecast_values((d[0],s[0],p[0]),frozen_cut)
    raw=predict(joint,x[i:i+1])[0]
    if available[i]:
        raw=.5*(raw+predict(joint,y[i:i+1])[0][[0,4,5,6,1,2,3]])
    raw=np.maximum(0.,raw)
    bins=raw[1:].reshape(2,3)
    side=int(np.argmax(bins.sum(1)))
    mass=bins[side].sum()
    estimate=float(bins[side]@centers/mass) if mass>0 else None
    alert=estimate is not None and mass-raw[0]>=cut
    update=bool(base["alert"] and alert and side+1==base["direction"])
    rows.append(dict(label=r,base=base,joint_alert=bool(alert),joint_direction=side+1,updated=update,
        reflected_available=bool(available[i]),distance_m=estimate if update else base["distance_m"]))
summary={}
for name in ("base","updated"):
    errors=[]
    bands={}
    count=Counter()
    for r in rows:
        if not r["base"]["alert"]:
            continue
        t=r["label"]
        if not t["direction"] or not 4<=t["distance"]<=20:
            count["out_of_band"]+=1
        elif t["direction"]==r["base"]["direction"]:
            pred=r["base"]["distance_m"] if name=="base" else r["distance_m"]
            error=abs(pred-t["distance"])
            errors.append(error)
            band="4-8" if t["distance"]<8 else "8-12" if t["distance"]<12 else "12-20"
            bands.setdefault(band,[]).append(error)
    summary[name]=dict(count,direction_hits=len(errors),within_5m=sum(e<=5 for e in errors),
        mae_m=mean(errors),by_band={k:dict(n=len(v),within_5m=sum(e<=5 for e in v),mae_m=mean(v))
                                    for k,v in bands.items()})
jsonl(out/"forecasts.jsonl",rows)
write(out/"forecasts-result.json",dict(summary=summary,n=len(rows),updated=sum(r["updated"] for r in rows),
    seed=None,termination="archive_end",revision=receipt["revision"],
    hashes={str(p):digest(p) for p in (labels,features,weights,fit_result,pins,Path(__file__))},
    rows_sha256=digest(out/"forecasts.jsonl")))
print(summary,"updates",sum(r["updated"] for r in rows))
