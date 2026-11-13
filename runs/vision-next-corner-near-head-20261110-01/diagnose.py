import json
from pathlib import Path
import numpy as np
from train_vision_teacher_line import digest,predict,read,write

out=Path(__file__).parent
source=Path("runs/vision-next-corner-20261110-03")
report=read(out/"result.json")
weights=out/"flyvis-weights.npz"
assert digest(weights)==report["models"]["flyvis"]["weights_sha256"]
rows=[json.loads(line) for line in (source/"rows.jsonl").read_text().splitlines()]
model_rows=[json.loads(line) for line in (out/"flyvis-predictions.jsonl").read_text().splitlines()]
by_key={(r["run"],r["frame"]):i for i,r in enumerate(rows)}
with np.load(weights,allow_pickle=False) as z:
    reg=tuple(z[f"dist_{k}"] for k in ("mean","scale","offset","coefficients"))
with np.load(source/"flyvis-features.npz",allow_pickle=False) as z:
    x=z["features"][[by_key[r["run"],r["frame"]] for r in model_rows]]
dist=np.array([r["distance"] if r["distance"] is not None else np.inf for r in model_rows])
role=np.array([r["split"] for r in model_rows])
near=np.array([r["direction"]!=0 for r in model_rows])&(dist<=30)
constant=float(np.median(dist[near&(role=="train")]))
prediction=predict(reg,x).clip(0.,30.)
result={}
for group in ("train","validation","heldout","test_other","external"):
    chosen=(role==group)&near
    close=chosen&(dist<=10.)
    result[group]=dict(n=int(chosen.sum()),close_n=int(close.sum()),
        constant_mae_m=float(np.mean(abs(constant-dist[chosen]))),
        flyvis_mae_m=float(np.mean(abs(prediction[chosen]-dist[chosen]))),
        close_constant_mae_m=float(np.mean(abs(constant-dist[close]))) if close.any() else None,
        close_flyvis_mae_m=float(np.mean(abs(prediction[close]-dist[close]))) if close.any() else None)
write(out/"conditional-distance.json",dict(constant_m=constant,results=result,source_sha256=digest(out/"result.json"),
    program_sha256=digest(Path(__file__)),scope="privileged near-event membership; no actionable detector output"))
print(result)
