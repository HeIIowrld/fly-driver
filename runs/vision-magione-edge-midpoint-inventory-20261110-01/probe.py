from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial import cKDTree
from audit_road_edge_projection import road_edges
from train_next_corner import BASE,MAGIONE,assign,CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import CameraArchive,at,digest,jsonl,read,write

out=Path(__file__).parent
previous=Path("runs/vision-corner-label-geometry-20261110-01")
dataset=Path("runs/vision-next-corner-20261110-03/dataset.json")
teacher.CORNER=CURVATURE_THRESHOLD
code=(previous/"probe.py").read_text(encoding="utf-8-sig")
exec(compile(code[code.index("def periodic_line("):code.index("def near(")],str(previous/"probe.py"),"exec"))
geometry=BASE/"heldout-magione-geometry-01/geometry.json"
assert digest(geometry)==read(dataset)["magione_geometry_sha256"]
edge_midpoint=np.mean(road_edges(read(geometry)["rows"]),axis=0)[:,[2,0]]
length,spans,stations,index=periodic_line(edge_midpoint)
samples=index.data
tangent=np.roll(samples,-1,axis=0)-np.roll(samples,1,axis=0)
tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
old={(r["run"],r["frame"]) for r in (json.loads(s) for s in
     (previous/"capture-labels.jsonl").read_text().splitlines())}
reports={}
rows=[]
for name,sub in (*MAGIONE,("magione-student-parent-01","run")):
    folder=BASE/name/sub/"vision"
    valid={}
    seen=Counter()
    with CameraArchive(folder) as archive:
        bundle=archive.bundle_sha256
        original=read(dataset)["sources"].get(name)
        if original:
            assert original["provenance"]["bundle_sha256"]==bundle
        for item in (json.loads(line) for line in archive.read("stream.jsonl").splitlines()):
            seen["raw"]+=1
            if item["ideal_line_shown"] or item["aid_probe_only"]:
                seen["aid"]+=1
                continue
            views={v["view_id"]:v for v in item["views"]}
            if set(views)!={"front","left","right","rear"}:
                seen["missing_view"]+=1
                continue
            frame=item["capture_id"]
            meta=archive.json(Path(item["file"]).name)
            assert meta["track_id"]=="magione" and meta["index"]==frame
            body=max(views.values(),key=lambda v:v["render_started_lower"])["source_render_metadata"]["source_body_after_render"]
            if not body["physics_available"] or body["speed_m_s"]<2.:
                seen["stationary_or_missing"]+=1
                continue
            pos=np.asarray(body["renderer_only_position"])[[2,0]]
            look=np.asarray(body["renderer_only_look"])[[2,0]]
            look/=np.linalg.norm(look)
            residual,i=index.query(pos)
            if residual>7 or tangent[i]@look<.7:
                seen["unregistered"]+=1
                continue
            assert frame not in valid
            valid[frame]=dict(station=float(stations[i]),time=item["received"],generation=item["generation"])
    for frame,r in valid.items():
        history=(frame-2,frame-1,frame)
        if any(i not in valid or valid[i]["generation"]!=r["generation"] for i in history):
            seen["missing_history"]+=1
            continue
        if not all(.045<valid[b]["time"]-valid[a]["time"]<.18 for a,b in zip(history,history[1:])):
            seen["cadence"]+=1
            continue
        label=assign(r["station"],spans,length,True)
        seen["eligible_history"]+=1
        if label is None:
            seen["already_inside"]+=1
            continue
        event,direction,distance=label
        if direction and 4<=distance<=20:
            seen["near"]+=1
            seen["near_le_10"]+=distance<=10
        seen["prior_inventory"]+=(name,frame) in old
        rows.append(dict(run=name,frame=frame,time=r["time"],event=event,direction=direction,
                         distance=float(distance) if direction else None,history=list(history),
                         in_prior_inventory=(name,frame) in old))
    reports[name]=dict(bundle_sha256=bundle,counts=dict(seen))
    print(name,dict(seen),flush=True)
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(revision=subprocess.check_output(["git","-c","safe.directory=*","rev-parse","HEAD"],text=True).strip(),
    seed=None,rows_sha256=digest(out/"rows.jsonl"),program_sha256=digest(Path(__file__)),
    geometry_sha256=digest(geometry),previous_source_sha256=digest(previous/"capture-comparison.json"),
    dataset_sha256=digest(dataset),spans=spans,reports=reports,
    scope="additional archived evaluation only; edge midpoint is a map proxy",
    termination="all five archived bundles screened"))
