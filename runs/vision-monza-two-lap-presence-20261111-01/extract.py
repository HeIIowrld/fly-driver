"""Export distance-training features with a physical-group-safe Monza split."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from extract_next_corner_flyvis import extract_pair, setup
from train_next_corner import BASE, MONZA
from train_vision_teacher_line import digest, jsonl, read, write

out=Path(__file__).parent
inventory=out.parent/"vision-monza-two-lap-corner-inventory-20261111-01"
old=out.parent/"vision-monza-midpoint-raw-head-20261110-01"
ref=out.parent/"vision-midpoint-reflected-retina-20261110-01"
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines()]
labels=inventory/"rows.jsonl"
receipt=read(inventory/"result.json")
assert digest(labels)==receipt["rows_sha256"]
selected=rows(labels)
assert len(selected)==receipt["n"]==989
old_rows=old/"rows.jsonl"
old_index={(r["run"],r["frame"]):i for i,r in enumerate(rows(old_rows))}
def archive_features(folder):
    path=folder/"features.npz"
    proof=read(folder/"flyvis-result.json")
    assert digest(path)==proof["features_sha256"] and digest(old_rows)==proof["rows_sha256"]
    with np.load(path,allow_pickle=False) as data:
        return data["features"],data["available"]
direct,available=archive_features(old)
mirror,ref_available=archive_features(ref)
torch.set_num_threads(1)
cv2.setNumThreads(1)
net,projector,indices,pool,step,model=setup(Path("runs/vision-next-corner-flyvis-cache-20261110"))
prior=read(Path("runs/vision-next-corner-20261110-03/flyvis-source.json"))
assert all(prior[k]==v for k,v in model.items())
run=MONZA[1][0]
needed={f for r in selected if r["run"]==run for f in r["history"]}
paired,bundle=extract_pair(BASE/run/"demonstration/camera/vision",needed,net,projector,indices,pool,step)
assert bundle==receipt["reports"][run]["provenance"]["camera_bundle_sha256"]
x=np.zeros((len(selected),1536),np.float32)
y=np.zeros_like(x)
usable=np.zeros(len(selected),bool)
for j,r in enumerate(selected):
    if r["run"]==run:
        x[j],y[j]=[np.concatenate([branch[i] for i in r["history"]]) for branch in paired]
        usable[j]=True
    else:
        i=old_index[r["run"],r["frame"]]
        assert available[i]
        x[j],y[j],usable[j]=direct[i],mirror[i],ref_available[i]
assert np.isfinite(x).all() and np.isfinite(y).all()
assert all(usable[i] for i,r in enumerate(selected) if r["split"]=="train")
jsonl(out/"rows.jsonl",selected)
np.savez_compressed(out/"features.npz",direct=x,reflected=y,reflected_available=usable)
write(out/"flyvis-result.json",dict(n=len(selected),reflected=int(usable.sum()),bundle_sha256=bundle,
    rows_sha256=digest(out/"rows.jsonl"),features_sha256=digest(out/"features.npz"),
    seed=None,revision=receipt["revision"],termination="archive_end",model=model,
    inputs_sha256={str(p):digest(p) for p in (labels,old_rows,old/"features.npz",ref/"features.npz",Path(__file__))}))
print(len(selected),int(usable.sum()),flush=True)
