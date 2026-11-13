from collections import Counter
from pathlib import Path
import json,subprocess

import cv2
import numpy as np
import train_vision_teacher_line as teacher
from train_next_corner import assign, CURVATURE_THRESHOLD, HISTORY_GUARD_M, COMPOUND_GAP_M
from train_vision_teacher_line import digest, jsonl, read, write

out=Path(__file__).parent
track=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
road=teacher.road_model(track)
_,grid,tree,length,holdouts=road
teacher.CORNER=CURVATURE_THRESHOLD
cv2.setNumThreads(1)
thermal=Path("runs/thermal-driving-20261103")
hot=Path("runs/thermal-control-improvement-20261104")
inputs=[
    (thermal/"cold40-demonstration-03",thermal/"cold40-qualification-01"),
    (thermal/"hot110-demonstration-01",thermal/"hot110-qualification-01"),
    (thermal/"nominal80-demonstration-01",thermal/"nominal80-qualification-01"),
    (hot/"hot110-demonstration-01",hot/"hot110-qualification-01"),
]
reports=[]
dataset=[]
features=[]
for run,qual in inputs:
    try:
        data,arc,_,prov=teacher.load_teacher(run,qual,recording=run)
        rows,quality=teacher.examples(run,qual,road,False,out,recording=run)
        spans=teacher.corners(arc,data[:,2:4])
        starts=np.array(grid[tree.query(teacher.at(arc,data[:,2:4],[s[0] for s in spans]))[1]])
        ends=np.array(grid[tree.query(teacher.at(arc,data[:,2:4],[s[1] for s in spans]))[1]])
        held=np.any(np.abs((starts[:,None]-holdouts+length/2)%length-length/2)<=90,axis=1)
        held|=np.any(np.abs((ends[:,None]-holdouts+length/2)%length-length/2)<=90,axis=1)
        groups=np.cumsum([0]+[int(b[0]-a[1]>COMPOUND_GAP_M) for a,b in zip(spans,spans[1:])])
        held=np.isin(groups,groups[held])
        candidates=[]
        candidate_features=[]
        for row in rows:
            info=assign(row["station"],spans,arc[-1])
            if info is None:
                continue
            event,direction,distance=info
            previous_end=spans[event-1][1] if event else 0.
            if row["station"]-previous_end<HISTORY_GUARD_M or not direction or distance>30:
                continue
            lo,hi=row["map_s"]-HISTORY_GUARD_M,row["map_s"]+distance
            if held[event] or any(lo<=h+90 and hi>=h-90 for h in holdouts):
                continue
            candidates.append(dict(run=str(run),frame=row["index"],time=row["t"],event=event,
                direction=direction,distance=distance,map_onset_m=float(starts[event]),
                history=row["history"],history_times=row["history_t"]))
            candidate_features.append(row["feature"])
        dataset.extend(candidates)
        features.extend(candidate_features)
        reports.append(dict(run=str(run),qualification_sha256=digest(qual/"result.json"),
            status="qualified",raw_rows=len(rows),near_n=len(candidates),
            near_le10=sum(r["distance"]<=10 for r in candidates),
            counts=dict(Counter(str((r["event"],r["direction"])) for r in candidates)),
            provenance=prov,quality=quality))
        print(run,len(rows),len(candidates),flush=True)
    except (ValueError, KeyError, FileNotFoundError, AssertionError) as error:
        reports.append(dict(run=str(run),status="rejected",error=type(error).__name__,reason=str(error)))
        print(run,repr(error),flush=True)
if dataset:
    jsonl(out/"rows.jsonl",dataset)
    np.save(out/"pixel-features.npy",np.stack(features),allow_pickle=False)
write(out/"result.json",dict(revision=subprocess.check_output(["git","-c","safe.directory=*","rev-parse","HEAD"],text=True).strip(),
    seed=None,program_sha256=digest(Path(__file__)),alignment_program_sha256=digest(Path(teacher.__file__)),
    track_sha256=digest(track),rows_sha256=digest(out/"rows.jsonl") if dataset else None,
    feature_sha256=digest(out/"pixel-features.npy") if dataset else None,reports=reports,
    termination="completed source eligibility and nearby approach extraction"))
