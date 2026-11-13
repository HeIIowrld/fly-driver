from pathlib import Path
from collections import Counter
import json,subprocess

import numpy as np
from scipy.spatial import cKDTree
from audit_road_edge_projection import road_edges
from train_next_corner import BASE,MONZA,assign,CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at,digest,read,write

out=Path(__file__).parent
dataset=Path("runs/vision-next-corner-20261110-03/dataset.json")
receipt=read(dataset)
teacher.CORNER=CURVATURE_THRESHOLD
monza=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
magione=BASE/"heldout-magione-geometry-01/geometry.json"
assert digest(monza)==receipt["monza_map_sha256"]
assert digest(magione)==receipt["magione_geometry_sha256"]

def periodic_line(xy):
    unique=np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-6]
    xy=xy[unique]
    if np.linalg.norm(xy[-1]-xy[0])<1e-6:
        xy=xy[:-1]
    closed=np.vstack((xy,xy[0]))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
    length=arc[-1]
    offset=int(np.ceil(length))
    spans=[(a-offset,b-offset,s) for a,b,s in teacher.corners(
        np.r_[arc[:-1]-length,arc[:-1],arc+length]+offset,np.vstack((xy,xy,closed)))
        if offset<=a<offset+length]
    samples=np.arange(0.,length)
    return length,spans,samples,cKDTree(at(arc,closed,samples))

def near(info):
    return info is not None and info[1]!=0 and 4<=info[2]<=20

def compare(left,right):
    counts=Counter()
    delta=[]
    for a,b in zip(left,right,strict=True):
        if near(a):
            counts["reference_near"]+=1
            if b is None:
                counts["midpoint_already_inside"]+=1
            elif not near(b):
                counts["midpoint_not_near"]+=1
            elif a[1]!=b[1]:
                counts["opposite_direction"]+=1
            else:
                counts["same_near_direction"]+=1
                delta.append(b[2]-a[2])
        if near(b):
            counts["midpoint_near"]+=1
    return dict(counts=dict(counts),same_near_distance_mae_m=float(np.mean(np.abs(delta))) if delta else None,
                signed_delta_quantiles_m=np.quantile(delta,[0,.5,1]).tolist() if delta else [])

data=np.genfromtxt(monza,names=True,delimiter=",")
midpoint=np.c_[(data["left_border_x"]+data["right_border_x"])/2,
               (data["left_border_y"]+data["right_border_y"])/2]
m_length,m_spans,m_grid,m_tree=periodic_line(midpoint)
monza_results=[]
for run,sub in MONZA:
    _,arc,_,_=teacher.load_teacher(BASE/run,BASE/run/sub)
    raw,arc,_,prov=teacher.load_teacher(BASE/run,BASE/run/sub)
    spans=teacher.corners(arc,raw[:,2:4])
    station=np.arange(1.,arc[-1]-30.)
    positions=at(arc,raw[:,2:4],station)
    residual,index=m_tree.query(positions)
    map_station=m_grid[index]
    increments=(np.diff(map_station)+m_length/2)%m_length-m_length/2
    if np.median(increments)<0:
        ml,ms,mg,mt=periodic_line(midpoint[::-1])
        residual,index=mt.query(positions)
        map_station=mg[index]
    else:
        ml,ms=m_length,m_spans
    keep=residual<=7
    rows=compare([assign(s,spans,arc[-1]) for s in station[keep]],
                 [assign(s,ms,ml,True) for s in map_station[keep]])
    monza_results.append(dict(run=run,station_n=int(keep.sum()),provenance=prov,**rows))
    print(run,rows,flush=True)

g=read(magione)["rows"]
ref=np.array([r["native_xyz"] for r in g])[:,[2,0]]
mid=np.mean(road_edges(g),axis=0)[:,[2,0]]
length,spans,station,tree=periodic_line(ref)
assert [list(s) for s in spans]==receipt["magione_map_corner_spans_m"]
ml,ms,mg,mt=periodic_line(mid)
closed=np.vstack((ref,ref[0]))
arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
residual,index=mt.query(at(arc,closed,station))
assert residual.max()<7.
reference_info=[assign(s,spans,length,True) for s in station]
midpoint_info=[assign(s,ms,ml,True) for s in mg[index]]
magione_result=dict(station_n=len(station),**compare(reference_info,midpoint_info),
                   reference_spans=spans,midpoint_spans=ms)
write(out/"result.json",dict(revision=subprocess.check_output(["git","-c","safe.directory=*","rev-parse","HEAD"],text=True).strip(),
    seed=None,inputs_sha256={str(p):digest(p) for p in (Path(__file__),monza,magione,dataset,
                   Path("scripts/train_vision_teacher_line.py"),Path("scripts/train_next_corner.py"))},
    monza=monza_results,magione=magione_result,scope="privileged label definitions only; no actor input",
    termination="completed source label comparison"))
print(magione_result,flush=True)
