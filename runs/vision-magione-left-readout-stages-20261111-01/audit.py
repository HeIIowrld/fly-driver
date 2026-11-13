"""Decompose frozen corner failures and direction evidence without refitting."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from flybrain_drive.flyvis_camera import VIEW_ORDER
from infer_frozen_next_corner import KEYS, contrast, load_features, load_readout
from train_vision_teacher_line import digest, jsonl, read, write

out=Path(__file__).parent
root=out.parent
frozen=root/"vision-frozen-next-corner-20261110-01"
pins=frozen/"pins.json"
index=frozen/"archive-parity/index.jsonl"
direct=frozen/"archive-parity/original/features.npz"
reflected=frozen/"archive-parity/reflected/features.npz"
cli=root/"vision-magione-joint-distance-fusion-20261111-01/cli/rows.jsonl"
truth=root/"vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl"
proof=read(cli.with_name("result.json"))
assert digest(cli)==proof["rows_sha256"]
assert digest(truth)==read(truth.with_name("result.json"))["rows_sha256"]
assert all(digest(path) in proof["inputs_sha256"].values() for path in (pins,index,direct,reflected))

def load(path):
    with path.open(encoding="utf8") as stream:
        return list(map(json.loads,stream))

def key(row):
    return row["run"],row["frame"]

records=load(index)
stored={key(r):r for r in load(cli)}
labels={key(r):r for r in load(truth)}
x,a,_=load_features(direct,digest(index),len(records))
y,b,_=load_features(reflected,digest(index),len(records))
assert a.all() and b.all()
spec=read(pins)
predict,threshold=load_readout(spec)
scores,ordinal,presence=predict(x,y,return_scores=True)
direct_scores,_,_=predict(x,None,return_scores=True)
mirror_scores,_,_=predict(None,y,return_scores=True)
assert np.allclose(.5*(direct_scores+mirror_scores),scores)

model=Path(spec["models"]["direction"]["path"])
with np.load(model,allow_pickle=False) as data:
    mean,scale,offset,coef=(data[f"cls_{k}"] for k in KEYS)
    bias=float(data["bias"])
x_parts=((contrast(x)-mean)/scale)[:,:,None]*coef
y_parts=((contrast(y)-mean)/scale)[:,:,None]*coef[:,[0,2,1]]
shape=(-1,3,len(VIEW_ORDER),8,16,3)
# Reflected left/right cameras correspond to the opposite original view.
parts=.5*(x_parts.reshape(shape).sum(axis=(1,3,4)) +
          y_parts.reshape(shape).sum(axis=(1,3,4))[:,[0,2,1,3],:])
intercept=.5*(offset+offset[[0,2,1]])+[bias,0.,0.]
assert np.allclose(parts.sum(axis=1)+intercept,scores,atol=1e-9)

def winner(values):
    return int(np.argmax(values)) if np.count_nonzero(values==max(values))==1 else 0

rows=[]
for i,record in enumerate(records):
    k=key(record)
    label=labels[k]
    actual=label["direction"]
    distance=label["distance"]
    pred=stored[k]
    raw=winner(scores[i])
    present=bool(presence[i]>=threshold)
    in_range=bool(ordinal[i]<=20.)
    assert raw==pred["direction"] and pred["alert"]==bool(raw and present and in_range)
    assert float(presence[i])==pred["presence_score"]
    near=bool(actual and distance is not None and 4.<=distance<=20.)
    # Crossed gates use identical presence and range, isolating the direction decision.
    branches={"paired":raw,"direct":winner(direct_scores[i]),"reflected":winner(mirror_scores[i]),
              "left_vs_right":int(np.argmax(scores[i,1:]))+1}
    branches.update({f"omit_{view}":winner(scores[i]-parts[i,j]) for j,view in enumerate(VIEW_ORDER)})
    rows.append(dict(run=k[0],frame=k[1],event=label["event"],actual=actual,
        onset_m=distance,near=near,raw=raw,present=present,in_range=in_range,
        ordinal_m=float(ordinal[i]),alert=pred["alert"],scores=scores[i].tolist(),
        left_minus_right_by_view=dict(zip(VIEW_ORDER,(parts[i,:,1]-parts[i,:,2]).tolist())),
        branches=branches))

summary={"threshold":threshold,"by_direction":{},"by_run":{},"counterfactual":{}}
for side in (1,2):
    subset=[r for r in rows if r["near"] and r["actual"]==side]
    summary["by_direction"][side]=dict(n=len(subset),
        raw_counts=dict(Counter(r["raw"] for r in subset)),
        raw_gate_counts={str(raw):dict(Counter(
            f"present_{int(r['present'])}_range_{int(r['in_range'])}" for r in subset if r["raw"]==raw))
            for raw in (0,1,2)},
        alerts=dict(Counter(r["raw"] for r in subset if r["alert"])),
        branch_agreement=dict(Counter(f"{r['branches']['direct']},{r['branches']['reflected']}" for r in subset)),
        median_left_minus_right_by_view={view:float(np.median([r["left_minus_right_by_view"][view] for r in subset]))
                                         for view in VIEW_ORDER})
for run in sorted({r["run"] for r in rows}):
    subset=[r for r in rows if r["run"]==run and r["near"]]
    summary["by_run"][run]=dict(n=len(subset),event=subset[0]["event"],actual=subset[0]["actual"],
        raw_counts=dict(Counter(r["raw"] for r in subset)),
        alerts=dict(Counter(r["raw"] for r in subset if r["alert"])))
for name in rows[0]["branches"]:
    active=[r for r in rows if r["branches"][name] and r["present"] and r["in_range"]]
    summary["counterfactual"][name]=dict(
        left_hits=sum(r["near"] and r["actual"]==r["branches"][name]==1 for r in active),
        right_hits=sum(r["near"] and r["actual"]==r["branches"][name]==2 for r in active),
        outside_alerts=sum(not r["near"] for r in active),
        far_or_no_corner_alerts=sum(not r["actual"] or r["onset_m"] is None or r["onset_m"]>30. for r in active))
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(summary=summary,seed=None,termination="archive_end",
    revision=spec["revision"],rows_sha256=digest(out/"rows.jsonl"),
    hashes={str(path):digest(path) for path in (pins,index,direct,reflected,cli,truth,model,Path(__file__))}))
print(summary)
