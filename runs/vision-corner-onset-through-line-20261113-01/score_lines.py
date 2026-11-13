"""Score the same Monza corner windows used in the earlier rolling-line audit."""
from collections import Counter, defaultdict
from pathlib import Path
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree

from dense_onset import ARCHIVE, OUT, RUNS, frames, old
from rolling_lines import LIMITS
import train_vision_teacher_line as teacher
from train_next_corner import CURVATURE_THRESHOLD

MAP=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
REFERENCE=RUNS/"vision-corner-label-geometry-20261110-01/probe.py"
PREVIOUS=RUNS/"vision-monza-rolling-cross-section-20261111-01"
WINDOWS=RUNS/"vision-monza-extension-width-guard-20261111-01"
PHASES={"approach","early_turn","mid_turn","late_turn","exit"}


def map_grid():
    teacher.CORNER=CURVATURE_THRESHOLD
    code=REFERENCE.read_text(encoding="utf-8-sig")
    scope=dict(np=np,teacher=teacher,cKDTree=cKDTree,at=teacher.at)
    exec(compile(code[code.index("def periodic_line("):code.index("def near(")],str(REFERENCE),"exec"),scope)
    data=np.genfromtxt(MAP,names=True,delimiter=",")
    xy=np.column_stack([(data[f"left_border_{a}"]+data[f"right_border_{a}"])/2 for a in ("x","y")])
    return scope["periodic_line"](xy)


def read_poses(expected):
    root=ARCHIVE/"monza-transfer-inputs"
    mapping=old.read(root/"archive-map.json")
    archive_proof=old.read(ARCHIVE/"monza/result.json")
    poses={}
    for run,path in mapping.items():
        with old.CameraArchive(path) as archive:
            assert archive.bundle_sha256==archive_proof["archive_bundles"][run]
            for raw in archive.read("stream.jsonl").splitlines():
                item=json.loads(raw)
                key=(run,item["capture_id"])
                if key not in expected:
                    continue
                meta=archive.json(Path(item["file"]).name)
                assert meta["index"]==key[1]
                pose=max(meta["views"],key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
                origin=np.asarray(pose["renderer_only_position"])[[2,0]]
                forward=np.asarray(pose["renderer_only_look"])[[2,0]]
                forward/=np.linalg.norm(forward)
                poses[key]=(origin,forward,item["received"],item["generation"])
    assert poses.keys()==expected
    return poses


def cross_sections(line, origin, forward, road, limits):
    length, _, station, tree = road
    start,end=limits
    normal=np.array([-forward[1],forward[0]])
    world=origin+np.asarray(line["x_m"])[:,None]*forward+np.asarray(line["y_m"])[:,None]*normal
    distance,nearest=tree.query(world)
    relative=(station[nearest]-start+length/2)%length-length/2
    if np.any(np.diff(relative)<0) or max(distance)>10.:
        return {},world
    points={}
    for s in range(int(min(relative)),int(max(relative))+1):
        i=int((start+s)%len(tree.data))
        tangent=tree.data[(i+1)%len(tree.data)]-tree.data[i-1]
        tangent/=np.linalg.norm(tangent)
        offset=world-tree.data[i]
        along=offset@tangent
        crossings=np.flatnonzero((along[:-1]<0)&(along[1:]>=0))
        if len(crossings)==1:
            j=crossings[0]
            point=offset[j]-along[j]*(offset[j+1]-offset[j])/(along[j+1]-along[j])
            points[s]=point@np.array([-tangent[1],tangent[0]])
    return points,world


def score(groups, lookup, poses, road, polygons, method):
    result=[]
    for (run,event), group in groups.items():
        start,end,_=road[1][event]
        previous=None
        chains=[]
        failures=Counter()
        safe=0
        for row in group:
            key=run,row["frame"]
            planned=lookup[key]["lines"][method]
            if planned is None:
                failures["no_line"]+=1
                previous=None
                continue
            origin,forward,t,generation=poses[key]
            assert (t,generation)==(row["received"],row["generation"])
            x,y,k=(np.asarray(planned[name]) for name in ("x_m","y_m","curvature_inv_m"))
            assert np.isfinite([x,y,k]).all() and np.all(np.diff(x)>0)
            assert np.max(abs(k))<=LIMITS["max_curvature"]+1.e-9
            points,world=cross_sections(planned,origin,forward,road,(start,end))
            outer,inner=polygons
            margin=min(min(cv2.pointPolygonTest(outer,tuple(p),True),
                           -cv2.pointPolygonTest(inner,tuple(p),True)) for p in world)
            if margin<LIMITS["clearance_m"] or not points:
                failures["map_clearance" if margin<LIMITS["clearance_m"] else "no_cross_section"]+=1
                previous=None
                continue
            safe+=1
            reason=None
            if previous is None:
                reason="first_available"
            else:
                f,t0,g,p0=previous
                common=points.keys()&p0.keys()
                if key[1]!=f+1 or g!=generation or not 0<t-t0<=.18:
                    reason="capture_gap"
                elif len(common)<5:
                    reason="overlap_under_5_stations"
                elif max(abs(points[s]-p0[s]) for s in common)>1:
                    reason="lateral_disagreement_over_1m"
            if reason:
                failures[reason]+=1
                chains.append(dict(phases=set(),frames=0,furthest=-np.inf))
            chain=chains[-1]
            chain["phases"].add(row["phase"])
            chain["frames"]+=1
            chain["furthest"]=max(chain["furthest"],max(points))
            previous=(key[1],t,generation,points)
        entries=[c for c in chains if "approach" in c["phases"]]
        result.append(dict(run=run,event=event,direction=int(road[1][event][2]),frames=len(group),bounded=safe,
                           whole_corner=any(c["phases"]==PHASES for c in chains),
                           reaches_exit=any(c["furthest"]>=end-start for c in entries),
                           longest=max((c["frames"] for c in chains),default=0),failures=dict(failures)))
    return result


def run():
    receipt=old.read(OUT/"line-result.json")
    path=OUT/"monza-lines.jsonl"
    lines=frames(path,receipt["tracks"]["monza"]["lines_sha256"])
    index={(r["run"],r["frame"]):r for r in lines}
    proof=old.read(WINDOWS/"result.json")
    source=WINDOWS/"rows.jsonl"
    rows=frames(source,proof["rows_sha256"])
    grouped=defaultdict(list)
    for row in rows:
        if row["policy"]=="short":
            grouped[row["run"],row["event"]].append(row)
    groups={key:sorted(value,key=lambda r:r["frame"]) for key,value in grouped.items()
            if {r["phase"] for r in value}==PHASES}
    assert len(groups)==22
    keys={(r["run"],r["frame"]) for rows in groups.values() for r in rows}
    assert keys.issubset(index)
    poses=read_poses(keys)
    road=map_grid()
    polygons=teacher.edge_polygons(MAP)
    output={}
    for method in ("classic","geometry","preview"):
        events=score(groups,index,poses,road,polygons,method)
        output[method]=dict(events=len(events),whole_corner=sum(e["whole_corner"] for e in events),
                           reaches_exit=sum(e["reaches_exit"] for e in events),bounded_frames=sum(e["bounded"] for e in events),
                           failures=sum((Counter(e["failures"]) for e in events),Counter()),detail=events)
        output[method]["by_direction"]={str(side):dict(events=sum(e["direction"]==side for e in events),
            whole=sum(e["whole_corner"] and e["direction"]==side for e in events)) for side in sorted({e["direction"] for e in events})}
        print(method,{k:v for k,v in output[method].items() if k!="detail"},flush=True)
    paths=(Path(__file__),MAP,REFERENCE,source,path,OUT/"line-result.json",
           PREVIOUS/"result.json",ARCHIVE/"monza-transfer-inputs/archive-map.json",ARCHIVE/"monza/result.json")
    old.write(OUT/"monza-through-corner.json",dict(methods=output,minimum_map_margin_m=LIMITS["clearance_m"],
              map_used_for_evaluation_only=True,min_overlap_stations=5,max_overlap_separation_m=1,
              source_sha256={str(p):old.digest(p) for p in paths},termination="archive_end"))


if __name__=="__main__":
    run()
