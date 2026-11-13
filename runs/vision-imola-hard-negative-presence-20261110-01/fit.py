import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from infer_frozen_next_corner import FEATURE_DIM, load_readout
from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
pins = Path("runs/vision-frozen-next-corner-20261110-01/pins.json")
predict, threshold = load_readout(read(pins))
paths = [pins,Path(__file__)]
def rows(path):
    return list(map(json.loads,path.read_text(encoding="utf8").splitlines()))

def invariant(a,b):
    direct = (a.astype(float)+b.astype(float))*.5
    grid = direct.reshape(-1,3,4,8,16)
    return (grid-grid.mean(-1,keepdims=True)).reshape(-1,FEATURE_DIM)

monza = Path("runs/vision-monza-midpoint-raw-head-20261110-01")
reflect = Path("runs/vision-midpoint-reflected-retina-20261110-01")
m = rows(monza/"rows.jsonl")
train = np.array([row["split"]=="train" for row in m])
for directory in (monza,reflect):
    proof = read(directory/"flyvis-result.json")
    assert digest(directory/"features.npz")==proof["features_sha256"]
    assert digest(monza/"rows.jsonl")==proof["rows_sha256"]
    paths.extend((directory/"features.npz",directory/"flyvis-result.json"))
with np.load(monza/"features.npz") as a,np.load(reflect/"features.npz") as b:
    assert b["available"].tolist()==train.tolist()
    anchors = invariant(a["features"][train],b["features"][train])

imola = rows(out/"imola-rows.jsonl")
proof=read(out/"result.json")
assert proof["feature_sha256"]==digest(out/"imola-features.npz")
assert proof["imola_rows_sha256"]==digest(out/"imola-rows.jsonl")
paths.extend((monza/"rows.jsonl",out/"imola-features.npz",out/"imola-rows.jsonl",out/"result.json"))
with np.load(out/"imola-features.npz") as f:
    a,b = f["features"].transpose(1,0,2)
x = invariant(a,b)
direction,distance,score = predict(a,b)
negative = np.array([r["run"]=="02" for r in imola])
assert negative.sum()==31 and (~negative).sum()==33
assert (distance<=20).all()
# One predeclared contrast: preserve the Monza score, target score zero for Imola negatives.
# Equal aggregate weight for the two sources; no tuning using the next Imola recording or Magione.
features = np.vstack((anchors,x[negative]))
target = np.r_[np.zeros(len(anchors)),-score[negative]]
weight = np.r_[np.full(len(anchors),.5/len(anchors)),np.full(negative.sum(),.5/negative.sum())]
weight *= len(weight)
center = np.average(features,axis=0,weights=weight)
scale = np.sqrt(np.average((features-center)**2,axis=0,weights=weight)).clip(.04)
design = (features-center)/scale
offset = np.average(target,weights=weight)
design *= np.sqrt(weight)[:,None]
gram=design@design.T
alpha=100.
gram.flat[::len(gram)+1] += alpha
coefficients=design.T@solve(gram,(target-offset)*np.sqrt(weight),assume_a="pos")
def adjusted(a,b,score):
    return score + offset + ((invariant(a,b)-center)/scale)@coefficients

summary = {}
new = adjusted(a,b,score)
for key, mask in (("imola_train",negative),("imola_adjacent_diagnostic",~negative)):
    summary[key] = dict(frames=int(mask.sum()),baseline=int((score[mask]>=threshold).sum()),
                        residual=int((new[mask]>=threshold).sum()))

root = Path("runs/vision-frozen-next-corner-20261110-01")
archive = root/"archive-parity"
label_path = Path("runs/vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl")
label_proof = label_path.with_name("result.json")
assert digest(label_path)==read(label_proof)["rows_sha256"]
forecast_proof = root/"archive-inference/result.json"
files=[archive/"index.jsonl",archive/"original/features.npz",archive/"reflected/features.npz"]
assert all(digest(p) in read(forecast_proof)["inputs_sha256"].values() for p in files)
paths.extend((label_path,label_proof,forecast_proof,*files))
index={(r["run"],r["frame"]):r for r in rows(label_path)}
truth=[index[r["run"],r["frame"]] for r in rows(files[0])]
with np.load(files[1]) as a,np.load(files[2]) as b:
    direct,reflected=a["features"],b["features"]
direction,distance,score=predict(direct,reflected)
new=adjusted(direct,reflected,score)
near=np.array([r["direction"] in (1,2) and r["distance"] is not None and 4<=r["distance"]<=20 for r in truth])
right=direction==np.array([r["direction"] for r in truth])
eligible=(direction!=0)&(distance<=20)
for mode,value in (("baseline",score),("residual",new)):
    alert=eligible&(value>=threshold)
    summary["magione_"+mode]=dict(near=int(near.sum()),direction_hits=int((near&right&alert).sum()),
                                  outside=int((~near&alert).sum()))
assert summary["magione_baseline"]==dict(near=152,direction_hits=66,outside=26)
np.savez_compressed(out/"candidate.npz",center=center,scale=scale,offset=offset,coefficients=coefficients)
write(out/"fit-result.json",dict(status="completed_presence_residual_development_test",alpha=alpha,
    threshold=threshold,summary=summary,monza_anchors=len(anchors),seed=None,
    candidate_sha256=digest(out/"candidate.npz"),hashes={str(p):digest(p) for p in paths},
    termination="archive_end",deployed=False,heldout_used=False))
print(summary)
