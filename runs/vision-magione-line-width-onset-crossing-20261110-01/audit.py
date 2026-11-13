"""Score saved body-frame paths against reference corner cross-sections."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from audit_road_edge_projection import road_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from train_next_corner import BASE, CURVATURE_THRESHOLD
from flybrain_drive.local_racing_line import plan_local_line
from plan_frozen_corner_archive import LIMITS
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at, digest, jsonl, read, write

out=Path(__file__).parent
root=out.parent/"vision-frozen-next-corner-20261110-01"
inventory=out.parent/"vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl"
paths=out.parent/"vision-magione-contrast-near-cli-20261110-01/rows.jsonl"
joint=out.parent/"vision-magione-corner-line-joint-audit-20261110-01/approach-rows.jsonl"
reference=out.parent/"vision-corner-label-geometry-20261110-01/probe.py"
geometry=BASE/"heldout-magione-geometry-01/geometry.json"
receipts=read(inventory.with_name("result.json"))
assert digest(reference.with_name("capture-comparison.json"))==receipts["previous_source_sha256"]
assert digest(geometry)==receipts["geometry_sha256"]
for p in (inventory,paths,joint):
    assert digest(p)==read(p.with_name("result.json"))["rows_sha256"]
teacher.CORNER=CURVATURE_THRESHOLD
code=reference.read_text(encoding="utf-8-sig")
exec(compile(code[code.index("def periodic_line("):code.index("def near(")],str(reference),"exec"))
midpoint=np.mean(road_edges(read(geometry)["rows"]),axis=0)[:,[2,0]]
length,spans,stations,tree=periodic_line(midpoint)
assert [list(s) for s in spans]==receipts["spans"]
samples=tree.data

def load(p):
    return {(r["run"],r["frame"]):r for r in map(json.loads,p.read_text(encoding="utf8").splitlines())}

labels=load(inventory)
lines=load(paths)
approaches=load(joint)
rows=[]
for run, folder in read(root/"archives.json").items():
    with CameraArchive(Path(folder)) as archive:
        assert archive.bundle_sha256==receipts["reports"][run]["bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            key=run,item["capture_id"]
            if key not in approaches:
                continue
            record=lines[key]
            line=record["line"]
            if line is None:
                continue
            meta=archive.json(Path(item["file"]).name)
            assert meta["index"]==key[1] and meta["track_id"]=="magione"
            pose=max(meta["views"],key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
            origin=np.asarray(pose["renderer_only_position"])[[2,0]]
            forward=np.asarray(pose["renderer_only_look"])[[2,0]]
            forward/=np.linalg.norm(forward)
            left=np.array([-forward[1],forward[0]])
            event=labels[key]["event"]
            start,end,_=spans[event]
            def crossing(station, world):
                i=station%len(samples)
                tangent=samples[(i+1)%len(samples)]-samples[i-1]
                signed=(world-samples[i])@tangent
                hits=np.flatnonzero((signed[:-1]<0)&(signed[1:]>=0))
                if len(hits)!=1:
                    return None
                j=hits[0]
                point=world[j]+(world[j+1]-world[j])*(-signed[j])/(signed[j+1]-signed[j])
                offset=float(np.linalg.norm(point-samples[i]))
                return offset if offset<=16. else None
            filtered=[]
            previous=None
            for section in record["sections"]:
                if section["valid"]:
                    width=section["left_y_m"]-section["right_y_m"]
                    if previous is not None and width>previous+1.5:
                        section=dict(section,valid=False)
                    previous=width
                filtered.append(section)
            limited,_=plan_local_line(filtered,record["body_speed_mps"],**LIMITS)
            assert limited is not None
            label=approaches[key]
            hit=(label["alert"] and label["forecast_direction"]==label["actual_direction"]
                 and abs(label["forecast_onset_m"]-label["actual_onset_m"])<=5.)
            options={}
            for name,candidate in (("base",line),("width_growth",limited)):
                world=origin+np.asarray(candidate["x_m"])[:,None]*forward+np.asarray(candidate["y_m"])[:,None]*left
                options[name]=dict(onset_crossing_offset_m=crossing(start,world),exit_crossing_offset_m=crossing(end,world),
                                   horizon_m=candidate["x_m"][-1])
            rows.append(dict(run=run,frame=key[1],event=event,forecast_hit=hit,options=options))
assert len(rows)==114
summary={}
for name,group in (("all_paths",rows),("correct_forecast",[r for r in rows if r["forecast_hit"]])):
    summary[name]={policy:dict(n=len(group),
        onset_crossing=sum(r["options"][policy]["onset_crossing_offset_m"] is not None for r in group),
        exit_crossing=sum(r["options"][policy]["exit_crossing_offset_m"] is not None for r in group))
        for policy in ("base","width_growth")}
    summary[name]["shortened"]=sum(r["options"]["base"]["horizon_m"]>r["options"]["width_growth"]["horizon_m"] for r in group)
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(summary=summary,seed=None,termination="archive_end",geometry_evaluation_only=True,
    revision=receipts["revision"], hashes={str(p):digest(p) for p in (inventory,paths,joint,reference,geometry,Path(__file__))},
    rows_sha256=digest(out/"rows.jsonl")))
print(summary)
