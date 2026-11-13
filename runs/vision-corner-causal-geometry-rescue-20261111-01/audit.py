"""Evaluate additive corner alerts with causal geometric confirmation."""
import json
from pathlib import Path
from statistics import median

import numpy as np
from infer_frozen_next_corner import load_readout
from train_vision_teacher_line import digest,jsonl,read,write

out=Path(__file__).parent
root=out.parent
causal=root/"vision-corner-front-bend-confirmation-20261111-01/causal-rows.jsonl"
monza=root/"vision-monza-corner-event-readout-20261111-01/rows.jsonl"
magione=root/"vision-magione-left-readout-stages-20261111-01/rows.jsonl"
forecast=root/"vision-magione-joint-distance-fusion-20261111-01/cli/rows.jsonl"
monza_forecast=root/"vision-monza-corner-guided-replay-20261111-01/forecasts.jsonl"
magione_index=root/"vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl"

def load(path,receipt=None):
    proof=read(receipt or path.with_name("result.json"))
    assert digest(path)==proof["rows_sha256"]
    return list(map(json.loads,path.read_text(encoding="utf8").splitlines()))

def key(r):
    return r["run"],r["frame"]

rows=load(causal,causal.with_name("causal-result.json"))
by_key={key(r):r for r in rows}
assert len(by_key)==len(rows)
readout={key(r):r for r in load(monza)+load(magione)}
original={key(r):r for r in load(forecast)}
original.update({key(r["label"]):r["base"] for r in
                 load(monza_forecast,monza_forecast.with_name("forecasts-result.json"))})
clock={key(r):r["time"] for r in load(monza)+load(magione_index)}
assert by_key.keys()==readout.keys()==original.keys()
data=root/"vision-monza-two-lap-presence-20261111-01"
features=data/"features.npz"
monza_index=data/"rows.jsonl"
receipt=read(data/"flyvis-result.json")
assert digest(features)==receipt["features_sha256"] and digest(monza_index)==receipt["rows_sha256"]
pins=root/"vision-frozen-next-corner-20261110-01/pins.json"
predict,_=load_readout(read(pins))
with np.load(features,allow_pickle=False) as packed:
    direct,reflected,available=(packed[k] for k in ("direct","reflected","reflected_available"))
for i, record in enumerate(map(json.loads,monza_index.read_text().splitlines())):
    _,distance,_=predict(direct[i:i+1],reflected[i:i+1] if available[i] else None)
    readout[key(record)]["ordinal_m"]=float(distance[0])

def bend_options(r):
    return [(x,1 if angle>0 else 2,abs(angle)) for x,angle in r["bends"]
            if 4.<=x<=20. and abs(angle)>=.025]

def options(current, prior):
    matches=[]
    for x,side,strength in bend_options(current):
        for previous,direction,earlier in bend_options(prior):
            if direction==side and -4.<=x-previous<=2.:
                matches.append(dict(direction=side,distance_m=x,strength=min(strength,earlier),
                                    prior_distance_m=previous))
    return sorted(matches,key=lambda r:(-r["strength"],abs(r["distance_m"]-r["prior_distance_m"]),
                                        r["distance_m"]))

outputs=[]
for r in rows:
    run,frame=key(r)
    paired=readout[run,frame]["scores"][1:]
    proposed=0 if paired[0]==paired[1] else 1+int(paired[1]>paired[0])
    ordinal=readout[run,frame]["ordinal_m"]
    prior=by_key.get((run,frame-1))
    valid_previous=prior is not None and prior["cohort"]==r["cohort"] and (
        0<clock[run,frame]-clock[run,frame-1]<=.24)
    candidates=options(r,prior) if valid_previous else []
    matching=[c for c in candidates if c["direction"]==proposed]
    distances=[c for c in matching if abs(c["distance_m"]-ordinal)<=5.]
    old=original[run,frame]
    frozen=dict(direction=old["direction"],distance_m=old["distance_m"]) if old["alert"] else None
    policies={"frozen":frozen}
    for name,pool in (("geometry",candidates),("matching_side",matching),("matching_distance",distances)):
        policies[name]=frozen or (pool[0] if pool else None)
    policies["matching_side_ordinal"]=frozen or (
        dict(matching[0],distance_m=ordinal) if matching and ordinal<=20. else None)
    outputs.append(dict(run=run,frame=frame,cohort=r["cohort"],actual=r["actual"],near=r["near"],
        truth_distance=readout[run,frame].get("onset_m",readout[run,frame].get("distance")),
        current_bends=bend_options(r), policies=policies))

def summarize(group,name):
    result=dict(left=0,right=0,within_5m=0,new_left=0,new_right=0,new_wrong=0,outside=0,far=0)
    for row in group:
        value=row["policies"][name]
        if value is None:
            continue
        if not row["near"]:
            result["outside"]+=1
            if not row["actual"] or row["truth_distance"] is None or row["truth_distance"]>30.:
                result["far"]+=1
        elif value["direction"]==row["actual"]:
            side="left" if row["actual"]==1 else "right"
            result[side]+=1
            result["new_"+side]+=row["policies"]["frozen"] is None
            result["within_5m"]+=abs(value["distance_m"]-row["truth_distance"])<=5.
        elif row["policies"]["frozen"] is None:
            result["new_wrong"]+=1
    return result

summary={name:{p:summarize([r for r in outputs if r["cohort"]==name],p) for p in outputs[0]["policies"]}
         for name in sorted({r["cohort"] for r in outputs})}
bias={}
for name in summary:
    group=[r for r in outputs if r["cohort"]==name and r["near"] and r["policies"]["frozen"] is None]
    for policy in ("geometry","matching_side_ordinal"):
        errors=[r["policies"][policy]["distance_m"]-r["truth_distance"] for r in group
                if r["policies"][policy] and r["policies"][policy]["direction"]==r["actual"]]
        if errors:
            bias.setdefault(name,{})[policy]=dict(n=len(errors),median_m=median(errors),
                                                 within_5m=sum(abs(e)<=5. for e in errors))
jsonl(out/"rows.jsonl",outputs)
write(out/"result.json",dict(summary=summary,new_distance_bias=bias,seed=None,termination="archive_end",
    revision=read(causal.with_name("causal-result.json"))["revision"],rows_sha256=digest(out/"rows.jsonl"),
    hashes={str(p):digest(p) for p in (causal,monza,magione,forecast,monza_forecast,magione_index,
                                     features,monza_index,pins,Path(__file__))}))
print(summary)
