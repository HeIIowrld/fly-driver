from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from train_next_corner import BASE,MONZA,assign,CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at,digest,jsonl,read,write

out=Path(__file__).parent
original=Path("runs/vision-next-corner-20261110-03")
teacher.CORNER=CURVATURE_THRESHOLD
base=Path("runs/vision-midpoint-aligned-head-20261110-01")
source=(base/"probe.py").read_text(encoding="utf-8-sig")
segment=source[source.index("mapfile="):source.index("all_rows=rows")]
prior=Path("runs/vision-corner-label-geometry-20261110-01/probe.py")
code=prior.read_text(encoding="utf-8-sig")
exec(code[code.index("def periodic_line("):code.index("def near(")])
metadata=read(original/"dataset.json")
exec(segment)
run,sub=MONZA[0]
rows,provenance=teacher.examples(BASE/run,BASE/run/sub,teacher.road_model(mapfile),False,out)
candidate=Counter()
selected=[]
validation_groups=read(base/"result.json")["validation_groups"]
for r in rows:
    residual,j=mt.query(r["xy"])
    station=float(mg[j])
    result=assign(station,ms,ml,True)
    if residual>7 or result is None:
        continue
    event,sign,distance=result
    group=int(groups[event])
    indexes=(np.arange(station-15.,station+min(distance,30.)+1).astype(int))%len(centers)
    if group in held or np.any(np.abs((centers[indexes,None]-holdouts+length/2)%length-length/2)<=90):
        continue
    if sign and distance<=30:
        candidate[str((group,sign))]+=1
    selected.append(dict(run=run,frame=r["index"],time=r["t"],history=r["history"],
                         split="validation" if group in validation_groups else "train",
                         physical_group=group,event=event,direction=sign,distance=float(distance) if sign else None))
assert sum(candidate.values())==read(base/"raw-monza-support.json")["near_n"]
jsonl(out/"rows.jsonl",selected)
write(out/"result.json",dict(revision=read(base/"result.json")["revision"],seed=None,
    provenance=provenance,near_groups=dict(candidate),rows_sha256=digest(out/"rows.jsonl"),
    validation_groups=validation_groups,termination="raw Monza midpoint labels exported",
    near_n=sum(candidate.values()),program_sha256=digest(Path(__file__)),source_sha256=digest(base/"result.json")))
print(candidate)
