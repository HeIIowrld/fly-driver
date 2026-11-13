import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import train_vision_teacher_line as teacher
from train_next_corner import BASE,CURVATURE_THRESHOLD
from train_vision_teacher_line import CameraArchive,at,digest,read,write

out=Path(__file__).parent
source=out/"rows.jsonl"
forecasts=out/"forecasts.jsonl"
assert digest(source)==read(out/"result.json")["rows_sha256"]
assert digest(forecasts)==read(out/"forecasts-result.json")["rows_sha256"]
def rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines()]
groups={}
for row in rows(source):
    if row["split"]=="validation":
        groups.setdefault((row["run"],row["frame"]),{})[row["policy"]]=row
changed={k:v for k,v in groups.items() if v["front_far"]["line"]!=v["guided"]["line"]}
labels={(r["label"]["run"],r["label"]["frame"]):r["label"] for r in rows(forecasts)}
teacher.CORNER=CURVATURE_THRESHOLD
ref=out.parent/"vision-corner-label-geometry-20261110-01/probe.py"
code=ref.read_text(encoding="utf-8-sig")
exec(code[code.index("def periodic_line("):code.index("def near(")])
track=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
assert digest(track) in read(out/"result.json")["hashes"].values()
d=np.genfromtxt(track,names=True,delimiter=",")
mid=np.c_[(d["left_border_x"]+d["right_border_x"])/2,(d["left_border_y"]+d["right_border_y"])/2]
_,spans,_,tree=periodic_line(mid)
center=tree.data
report=[]
for (run,frame),policies in changed.items():
    with CameraArchive(BASE/run/"demonstration/camera/vision") as archive:
        item=next(r for l in archive.read("stream.jsonl").splitlines() if (r:=json.loads(l))["capture_id"]==frame)
        meta=archive.json(Path(item["file"]).name)
        pose=max(meta["views"],key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
        origin=np.asarray(pose["renderer_only_position"])[[2,0]]
        forward=np.asarray(pose["renderer_only_look"])[[2,0]]
        forward/=np.linalg.norm(forward)
        left=np.array([-forward[1],forward[0]])
    label=labels[run,frame]
    start=spans[label["event"]][0]
    result={}
    for name,r in policies.items():
        line=r["line"]
        world=origin+np.asarray(line["x_m"])[:,None]*forward+np.asarray(line["y_m"])[:,None]*left
        result[name]={}
        for shift in (0,4):
            i=(start+shift)%len(center)
            tangent=center[(i+1)%len(center)]-center[i-1]
            signed=(world-center[i])@tangent
            indices=np.flatnonzero((signed[:-1]<0)&(signed[1:]>=0))
            result[name][shift]=bool(len(indices)==1)
    report.append(dict(run=run,frame=frame,upcoming_event=label["event"],
                       actual_distance_m=label["distance"],crossings=result))
write(out/"validation-onset.json",dict(rows=report,inputs_sha256={str(p):digest(p) for p in
    (source,forecasts,track,ref,Path(__file__))}))
print(report)
