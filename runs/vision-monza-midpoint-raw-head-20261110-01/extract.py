from pathlib import Path
import json,time
import cv2
import numpy as np
import torch
from extract_next_corner_flyvis import setup, extract
from train_vision_teacher_line import digest, read, write

out=Path(__file__).parent
inventory=out
receipt=read(inventory/"result.json")
assert digest(inventory/"rows.jsonl")==receipt["rows_sha256"]
rows=[json.loads(line) for line in (inventory/"rows.jsonl").read_text().splitlines()]
torch.set_num_threads(1)
cv2.setNumThreads(1)
net,projector,indices,pool,step,source=setup(Path("runs/vision-next-corner-flyvis-cache-20261110"))
previous=read(Path("runs/vision-next-corner-20261110-03/flyvis-source.json"))
assert all(previous[k]==v for k,v in source.items())
x=np.zeros((len(rows),1536),dtype=np.float32)
available=np.zeros(len(rows),dtype=bool)
reports={}
start=time.monotonic()
from train_next_corner import BASE,MONZA
run,_=MONZA[0]
subset=[(i,r["history"]) for i,r in enumerate(rows)]
result,bundle,delays=extract(BASE/run/"demonstration/camera/vision",{f for _,h in subset for f in h},
                             net,projector,indices,pool,step,"last-render")
assert bundle==receipt["provenance"]["camera_bundle_sha256"]
for i,history in subset:
    if all(f in result for f in history):
        x[i]=np.concatenate([result[f] for f in history])
        available[i]=True
reports[run]=dict(bundle_sha256=bundle,requested=len(subset),available=int(available.sum()))
print(reports,flush=True)
assert np.isfinite(x).all() and available.any()
np.savez_compressed(out/"features.npz",features=x,available=available)
write(out/"flyvis-result.json",dict(**source,revision=receipt["revision"],seed=None,
    inventory_sha256=digest(inventory/"result.json"),rows_sha256=receipt["rows_sha256"],
    program_sha256=digest(Path(__file__)),extractor_sha256=digest(Path("scripts/extract_next_corner_flyvis.py")),
    features_sha256=digest(out/"features.npz"),reports=reports,elapsed_s=time.monotonic()-start,
    read_at="last-render",optimizer_updates=0,termination="completed"))
