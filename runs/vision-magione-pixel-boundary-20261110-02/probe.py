import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from audit_road_edge_projection import road_edges, section_points
from flybrain_drive.local_racing_line import plan_local_line
from train_next_corner import BASE, MAGIONE
from train_vision_teacher_line import CameraArchive, VIEWS, digest, jsonl, read, write

OUT=Path("runs/vision-magione-pixel-boundary-20261110-02")
CAMERA=Path("configs/ac-csp-exterior-front-v3-retina.json")
CANDIDATES=Path("runs/vision-next-corner-20261110-03/rows.jsonl")
GEOMETRY=BASE/"heldout-magione-geometry-01/geometry.json"
X=np.array([10.,15.,20.])
Y=np.linspace(-14.,14.,281)
points=np.stack(np.meshgrid(X,Y,indexing="ij"),axis=-1)
points=np.concatenate((points,np.zeros((*points.shape[:2],1))),axis=2)
limits=dict(clearance_m=1.1,max_curvature=.15,lateral_accel_mps2=4.,braking_mps2=4.,speed_ceiling_mps=25.)

def sampling(config):
    lookups={}
    for name in VIEWS:
        c=config["views"][name]
        delta=points-c["origin_body_m"]
        z=delta@c["forward_body"]
        safe=np.where(z>0,z,1.)
        u=.5+.5*c["projection_x"]*(delta@c["right_body"])/safe
        v=.5-.5*c["projection_y"]*(delta@c["up_body"])/safe
        valid=(z>0)&(u>=.02)&(u<=.98)&(v>=.02)&(v<=.98)
        lookups[name]=(u,v,valid)
    return lookups

def boundary(images,luts):
    # Classify local vegetation; a missing side remains explicitly invalid.
    # No world pose, map, record-specific projection matrices, or teacher input.
    g_votes=np.zeros(points.shape[:2],dtype=int)
    seen=np.zeros_like(g_votes)
    for name in VIEWS:
        u,v,ok=luts[name]
        image=images[name]
        height,width=image.shape[:2]
        b,g,r=image[np.rint(v[ok]*(height-1)).astype(int),np.rint(u[ok]*(width-1)).astype(int)].astype(float).T
        vegetation=(g-r>6)&(g-b>12)&(g>40)
        g_votes[ok]+=vegetation
        seen[ok]+=1
    grass=(g_votes>seen/2)&(seen>0)
    sections=[]
    center=0.
    for j,x in enumerate(X):
        # Confirm vegetation and pavement on each side over a short distance.
        kernel=np.ones(5)
        veg=np.convolve(grass[j],kernel,"same")>=4
        pavement=np.convolve((seen[j]>0)&~grass[j],kernel,"same")>=4
        supported=(seen[j]>0)
        start=int(np.argmin(abs(Y-center)))
        left=right=None
        if pavement[start]:
            gap=0
            for q in range(start-1,5,-1):
                gap=0 if supported[q] else gap+1
                if gap>2:
                    break
                if gap:
                    continue
                if veg[q] and pavement[q+3:q+8].sum()>=3:
                    right=float(Y[q])
                    break
            gap=0
            for q in range(start+1,len(Y)-6):
                gap=0 if supported[q] else gap+1
                if gap>2:
                    break
                if gap:
                    continue
                if veg[q] and pavement[q-7:q-2].sum()>=3:
                    left=float(Y[q])
                    break
        ok=left is not None and right is not None and 3.<left-right<20.
        sections.append(dict(forward_x_m=float(x),right_y_m=right,left_y_m=left,valid=ok))
        if ok:
            center=(right+left)/2
    return sections

def evaluate(episode,sub,ids,luts,edges):
    rows=[]
    directory=BASE/episode/sub/"vision"
    with CameraArchive(directory) as archive:
        for line in archive.read("stream.jsonl").splitlines():
            item=json.loads(line)
            frame=item["capture_id"]
            if frame not in ids:
                continue
            assert not item["ideal_line_shown"] and not item["aid_probe_only"]
            views={v["view_id"]:v for v in item["views"]}
            assert set(views)==set(VIEWS)
            images={n:cv2.imdecode(np.frombuffer(archive.read(Path(views[n]["file"]).name),np.uint8),cv2.IMREAD_COLOR)
                    for n in VIEWS}
            assert all(im is not None and im.shape==(180,320,3) for im in images.values())
            sections=boundary(images,luts)
            # Diagnostic map proxy only. This is not an annotated pavement edge.
            last=max(views.values(),key=lambda v:v["render_started_lower"])
            body=last["source_render_metadata"]["source_body_after_render"]
            truth=section_points(edges,body)
            origin=np.array(body["renderer_only_position"])
            look=np.array(body["renderer_only_look"])[[2,0]]
            look/=np.linalg.norm(look)
            lateral=(truth-origin)[:,:,[2,0]]@np.array([-look[1],look[0]])
            error=[(np.abs(lateral[j,::-1]-[s["right_y_m"],s["left_y_m"]]).tolist()
                    if s["valid"] and np.isfinite(lateral[j]).all() else None) for j,s in enumerate(sections)]
            line_out,status=plan_local_line(sections,0.,**limits)
            rows.append(dict(frame=frame,sections=sections,map_proxy_abs_error_m=error,
                             path_generated=line_out is not None,status=status))
        bundle=archive.bundle_sha256
    assert len(rows)==len(ids)
    jsonl(OUT/f"{episode}.jsonl",rows)
    return rows,bundle

cv2.setNumThreads(1)
assert digest(GEOMETRY)==read(GEOMETRY.parent/"result.json")["geometry_sha256"]
edges=road_edges(read(GEOMETRY)["rows"])
lookup=sampling(read(CAMERA))
assert all(np.isfinite(u).all() and np.isfinite(v).all() for u,v,_ in lookup.values())
source_rows=[json.loads(s) for s in CANDIDATES.read_text().splitlines()]
all_rows=[]
sources={}
for episode,sub in MAGIONE:
    ids={r["frame"] for r in source_rows if r["run"]==episode and r["split"]=="external"}
    rows,bundle=evaluate(episode,sub,ids,lookup,edges)
    all_rows.extend(rows)
    sources[episode]=dict(bundle_sha256=bundle,n=len(rows),output_sha256=digest(OUT/f"{episode}.jsonl"))
    print(episode,len(rows),Counter(r["status"] for r in rows),flush=True)
summary=[]
for j,station in enumerate(X):
    compared=[r["map_proxy_abs_error_m"][j] for r in all_rows if r["map_proxy_abs_error_m"][j] is not None]
    summary.append(dict(distance_m=float(station),valid=sum(r["sections"][j]["valid"] for r in all_rows),
        compared=len(compared),map_proxy_mae_m=float(np.mean(compared)) if compared else None))
write(OUT/"result.json",dict(revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",seed=None,
    inputs_sha256={str(p):digest(p) for p in (CAMERA,CANDIDATES,GEOMETRY,Path(__file__),Path("scripts/audit_road_edge_projection.py"))},
    sources=sources,summary=summary,path_generated=sum(r["path_generated"] for r in all_rows),
    scope="stored actual AC pixels; ground plane and vegetation heuristic; map approximation for evaluation only",
    termination="all archived Magione captures evaluated"))
print(summary)
