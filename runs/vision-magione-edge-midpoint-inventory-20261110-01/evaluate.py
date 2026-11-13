from pathlib import Path
from collections import Counter
import json
import numpy as np
from train_vision_teacher_line import digest,jsonl,predict,read,write

out=Path(__file__).parent
inventory=read(out/"result.json")
features=read(out/"flyvis-result.json")
assert digest(out/"rows.jsonl")==inventory["rows_sha256"]==features["rows_sha256"]
assert digest(out/"features.npz")==features["features_sha256"]
rows=[json.loads(line) for line in (out/"rows.jsonl").read_text().splitlines()]
with np.load(out/"features.npz",allow_pickle=False) as z:
    x=z["features"][z["available"]].astype(float)
    rows=[r for r,m in zip(rows,z["available"],strict=True) if m]
previous=Path("runs/vision-next-corner-20261110-03")
original=read(previous/"result.json")
assert digest(previous/"flyvis-features.npz")==original["flyvis_features_sha256"]
assert digest(previous/"rows.jsonl")==original["rows_sha256"]
old_rows=[json.loads(l) for l in (previous/"rows.jsonl").read_text().splitlines()]
lookup={(r["run"],r["frame"]):i for i,r in enumerate(old_rows)}
overlap=[(i,lookup[r["run"],r["frame"]]) for i,r in enumerate(rows) if r["in_prior_inventory"]]
with np.load(previous/"flyvis-features.npz",allow_pickle=False) as original_features:
    a,b=np.asarray(overlap).T
    replay_max_error=float(np.max(abs(x[a]-original_features["features"][b])))
truth=np.array([r["direction"] for r in rows])
distance=np.array([r["distance"] if r["distance"] is not None else np.inf for r in rows])
near=(truth!=0)&(distance>=4)&(distance<=20)
close=near&(distance<=10)
early=(truth!=0)&(distance>20)&(distance<=30)
models=[
    ("long_120m",Path("runs/vision-next-corner-20261110-03/flyvis-weights.npz"),120.),
    ("near_30m_original",Path("runs/vision-next-corner-near-head-20261110-01/flyvis-weights.npz"),30.),
    ("near_30m_augmented",Path("runs/vision-monza-near-flyvis-20261110-01/augmented-weights.npz"),30.),
]
assert digest(models[0][1])==read(models[0][1].parent/"comparison-sha256.json")["flyvis-weights.npz"]
assert digest(models[1][1])==read(models[1][1].parent/"result.json")["models"]["flyvis"]["weights_sha256"]
assert digest(models[2][1])==read(models[2][1].parent/"readout-result.json")["models"]["augmented"]["weights_sha256"]
metrics={}
for name,path,horizon in models:
    with np.load(path,allow_pickle=False) as z:
        keys=("mean","scale","offset","coefficients")
        classifier=tuple(z[f"cls_{k}"] for k in keys)
        regressor=tuple(z[f"dist_{k}"] for k in keys)
        bias=float(z["bias"])
    category=(predict(classifier,x)+[bias,0.,0.]).argmax(1)
    predicted_distance=predict(regressor,x).clip(0.,horizon)
    alert=(category!=0)&(predicted_distance<=20.)
    hit=near&(category==truth)
    timely=hit&alert
    errors=abs(predicted_distance-distance)
    metrics[name]=dict(weights_sha256=digest(path),frames=len(rows),near=int(near.sum()),
        near_direction_hit=int(hit.sum()),near_direction_and_alert=int(timely.sum()),
        near_correct_within_5m=int((timely&(errors<=5)).sum()),near_le10_n=int(close.sum()),
        near_le10_correct_within_5m=int((timely&close&(errors<=5)).sum()),
        alert_outside_4_20=int((alert&~near).sum()),
        alert_over_30m_or_none=int((alert&(distance>30)).sum()),
        alert_at_20_30m=int((alert&early).sum()),
        minimum_emitted_distance_m=float(predicted_distance[category!=0].min()) if (category!=0).any() else None)
    jsonl(out/f"{name}-predictions.jsonl",(
        dict(r,predicted=int(c),predicted_distance_m=float(d) if c else None)
        for r,c,d in zip(rows,category,predicted_distance,strict=True)))
write(out/"fixed-head-evaluation.json",dict(revision=inventory["revision"],seed=None,
    metrics=metrics,overlap_frames=len(overlap),overlap_max_error=replay_max_error,near_by_run=dict(Counter(r["run"] for r,m in zip(rows,near,strict=True) if m)),
    source_sha256=digest(out/"flyvis-result.json"),program_sha256=digest(Path(__file__)),
    scope="new external road-midpoint labels; frozen prior heads; no fit on Magione",termination="completed"))
print(metrics)
