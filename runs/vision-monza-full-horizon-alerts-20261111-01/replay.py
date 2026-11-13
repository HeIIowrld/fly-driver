"""Replay frozen corner forecasts for every previously scored Monza path frame."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from extract_next_corner_flyvis import extract_pair,setup
from infer_frozen_next_corner import forecast_values,load_joint_distances,load_readout
from train_next_corner import BASE,MONZA
from train_vision_teacher_line import digest,jsonl,read,write

out=Path(__file__).parent
root=out.parent
source=root/"vision-monza-guided-long-horizon-20261111-01/rows.jsonl"
proof=read(source.with_name("result.json"))
assert digest(source)==proof["rows_sha256"]
evaluated=[r for l in source.read_text(encoding="utf8").splitlines()
           if (r:=json.loads(l))["policy"]=="short"]
assert len(evaluated)==1318
pins=root/"vision-frozen-next-corner-20261110-01/pins.json"
joint_pins=root/"vision-magione-joint-distance-fusion-20261111-01/joint-pins.json"
predict,threshold=load_readout(read(pins))
joint,model_path=load_joint_distances(read(joint_pins))
torch.set_num_threads(1)
cv2.setNumThreads(1)
net,projector,indices,pool,step,model=setup(Path("runs/vision-next-corner-flyvis-cache-20261110"))
prior=read(root/"vision-next-corner-20261110-03/flyvis-source.json")
assert all(prior[k]==v for k,v in model.items())
rows=[]
reports={}
inputs=[source,pins,joint_pins,model_path,Path(__file__)]
for run,qualification in MONZA:
    frames=[r["frame"] for r in evaluated if r["run"]==run]
    required={i for frame in frames for i in range(max(0,frame-2),frame+1)}
    clocks={}
    def on_frame(row,features,images):
        clocks[row["capture_id"]]=(row["generation"],row["received"])
    print(run,"extracting",len(required),"retinal snapshots",flush=True)
    a,b=BASE/run,BASE/run/qualification/"result.json"
    inputs.append(b)
    outputs,bundle=extract_pair(a/"demonstration/camera/vision",required,
        net,projector,indices,pool,step,on_frame=on_frame)
    assert bundle==read(b)["camera_bundle_sha256"]
    completed=[]
    direct,mirror=[],[]
    for frame in frames:
        history=(frame-2,frame-1,frame)
        if not all(i in clocks and all(i in branch for branch in outputs) for i in history):
            continue
        samples=[clocks[i] for i in history]
        times=np.diff([t for _,t in samples])
        if len({g for g,_ in samples})!=1 or not ((.045<times)&(times<.18)).all():
            continue
        a,b=[np.concatenate([branch[i] for i in history]) for branch in outputs]
        direct.append(a)
        mirror.append(b)
        completed.append(frame)
    x,y=map(np.asarray,(direct,mirror))
    raw=predict(x,y)
    j_side,j_dist,j_alert=joint(x,y)
    predicted={}
    for f,d,s,p,jd,js,ja in zip(completed,*raw,j_side,j_dist,j_alert,strict=True):
        result=forecast_values((d,s,p),threshold)
        if result["alert"] and ja and jd==result["direction"]:
            result["distance_m"]=float(js)
        predicted[f]=result
    rows.extend(dict(run=run,frame=f,**predicted.get(f,forecast_values(None,threshold))) for f in frames)
    reports[run]=dict(n=len(frames),available=len(completed),bundle_sha256=bundle)
    print(run,reports[run],flush=True)
assert len(rows)==len(evaluated)
jsonl(out/"forecasts.jsonl",rows)
write(out/"forecast-result.json",dict(reports=reports,seed=None,termination="archive_end",
    revision=proof["revision"],model=model,rows_sha256=digest(out/"forecasts.jsonl"),
    inputs_sha256={str(p):digest(p) for p in inputs}))
