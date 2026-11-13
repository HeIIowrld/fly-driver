"""Grouped next-corner learning with paired fixed-vision comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path

import train_vision_teacher_line as teacher_line

import cv2
import numpy as np
from scipy.spatial import cKDTree

from train_vision_road_corridor import external_rows
from train_vision_teacher_line import (
    at, check, corners, digest, examples, fit, load_teacher, predict, read, road_model, write, jsonl,
)

BASE = Path("runs/braking-corner-learning-20261105")
MONZA = (("monza-user-started-teacher-01","qualification-03"),("fresh-monza-develop-01","qualification-02"))
MAGIONE = (("magione-student-parent-02","run"),("magione-student-candidate-01","run"),
           ("magione-user-recovered-01","candidate"),("magione-alternative-student-parent-01","run"))
CLASSES = ("none","left","right")
HORIZON = 120.
HISTORY_GUARD_M = 15.
CURVATURE_THRESHOLD = .002
COMPOUND_GAP_M = 20.


def assign(station, spans, length, periodic=False):
    if periodic and spans[-1][1]>length and station<spans[-1][1]-length:
        return None
    for i,(start,end,sign) in enumerate(spans):
        if start<=station<end:
            return None
        if start>station:
            break
    else:
        if not periodic:
            return None
        i=0
        start,_,sign=spans[0]
        start+=length
    distance=start-station
    return i, int(1 if sign>0 else 2) if distance<=HORIZON else 0, distance


def extract(out):
    cv2.setNumThreads(1)
    teacher_line.CORNER = CURVATURE_THRESHOLD
    track=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
    road=road_model(track)
    _,grid,tree,length,holdouts=road
    datasets={}
    sources={}
    for name,qualification in MONZA:
        run=BASE/name
        rows,prov=examples(run,run/qualification,road,False,out)
        data,arc,_,_=load_teacher(run,run/qualification)
        spans=corners(arc,data[:,2:4])
        # Nearby consecutive curvature pieces in a chicane stay together.
        event_map=np.array(grid[tree.query(at(arc,data[:,2:4],[s[0] for s in spans]))[1]])
        event_end=np.array(grid[tree.query(at(arc,data[:,2:4],[s[1] for s in spans]))[1]])
        held=np.any(np.abs((event_map[:,None]-holdouts+length/2)%length-length/2)<=90,axis=1)
        held|=np.any(np.abs((event_end[:,None]-holdouts+length/2)%length-length/2)<=90,axis=1)
        event_group = np.cumsum([0]+[int(b[0]-a[1]>COMPOUND_GAP_M) for a,b in zip(spans,spans[1:])])
        held = np.isin(event_group,event_group[held])
        usable=[]
        previous_ends=np.r_[0.,[s[1] for s in spans[:-1]]]
        for r in rows:
            info=assign(r["station"],spans,arc[-1])
            if info is None:
                continue
            i,category,distance=info
            if r["station"]-previous_ends[i]<HISTORY_GUARD_M:
                continue
            if not category and r["station"]+HORIZON>arc[-1]:
                continue
            usable.append((r,i,category,distance))
        is_develop=name==MONZA[0][0]
        selected=[]
        if is_develop:
            for cat in (1,2):
                candidates=sorted({i for _,i,c,_ in usable if c==cat and not held[i]})
                check(len(candidates)>=2,f"need two distinct eligible {CLASSES[cat]} events")
                ordered = candidates if cat==1 else list(reversed(candidates))
                chosen = next(i for i in ordered if event_group[i] not in selected)
                selected.append(int(event_group[chosen]))
        records=[]
        for r,i,c,dist in usable:
            split=("heldout" if held[i] else "test_other") if not is_develop else (
                "excluded" if held[i] else "validation" if event_group[i] in selected else "train")
            # Protect the original whole map holdout including the visual history.
            if is_develop and split in ("train","validation"):
                limits=(r["map_s"]-HISTORY_GUARD_M,r["map_s"]+min(HORIZON,dist))
                if any(limits[0]<=h+90 and limits[1]>=h-90 for h in holdouts):
                    split="excluded"
            records.append((r,i,c,dist,split))
        datasets[name]=records
        sources[name]=dict(provenance=prov,corner_spans_m=spans,corner_map_m=event_map.tolist(),
                           heldout_events=np.flatnonzero(held).tolist(),validation_events=[i for i,g in enumerate(event_group) if g in selected], event_groups=event_group.tolist())
    geo=BASE/"heldout-magione-geometry-01/geometry.json"
    g=read(geo)
    check(read(geo.parent/"result.json")["geometry_sha256"]==digest(geo),"geometry receipt")
    xy=np.array([r["native_xyz"] for r in g["rows"]])[:,[2,0]]
    xy=np.vstack([xy,xy[0]])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    grid=np.arange(0.,arc[-1],1.)
    center=at(arc,xy,grid)
    index=cKDTree(center)
    tangent=np.roll(center,-1,axis=0)-np.roll(center,1,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    # Pad the periodic geometry before smoothing, then retain one lap of starts.
    padded_arc=np.r_[arc[:-1]-arc[-1],arc[:-1],arc+arc[-1]]
    padded_xy=np.concatenate([xy[:-1],xy[:-1],xy])
    offset=int(np.ceil(arc[-1]))
    check(np.all(np.diff(padded_arc)>0),"periodic arc ordering")
    spans=[(a-offset,b-offset,sgn) for a,b,sgn in corners(padded_arc+offset,padded_xy)
           if offset<=a<offset+arc[-1]]
    for name,sub in MAGIONE:
        rows,prov=external_rows(BASE/name/sub/"vision")
        records=[]
        for r in rows:
            distance,i=index.query(r["xy"])
            if distance>7. or tangent[i]@r["rot"][:,0]<.7:
                continue
            info=assign(float(grid[i]),spans,arc[-1],True)
            if info is None:
                continue
            event,c,dist=info
            records.append((r,event,c,dist,"external"))
        if records:
            datasets[name]=records
        sources[name]=dict(provenance=prov,registered_pre_entry=len(records))
    records=[]
    feats=[]
    for name,items in datasets.items():
        for r,i,c,d,split in items:
            records.append(dict(run=name,frame=r["index"],time=r["t"],split=split,event=i,
                                direction=c,distance=float(d) if c else None))
            feats.append(r["feature"])
    for split in ("train","validation"):
        n=Counter(r["direction"] for r in records if r["split"]==split)
        check(set(n)=={0,1,2},f"missing class in {split}: {n}")
    np.save(out/"image-features.npy",np.stack(feats),allow_pickle=False)
    jsonl(out/"rows.jsonl",records)
    write(out/"dataset.json",dict(sources=sources,monza_map_sha256=digest(track),magione_geometry_sha256=digest(geo),
          magione_map_corner_spans_m=spans,history_guard_m=HISTORY_GUARD_M,horizon_m=HORIZON,curvature_threshold_inv_m=CURVATURE_THRESHOLD,compound_gap_m=COMPOUND_GAP_M))


def macro(truth,pred):
    return float(np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))


def select_model(x,rows):
    y=np.array([r["direction"] for r in rows])
    d=np.array([r["distance"] or 0. for r in rows])
    train=np.array([r["split"]=="train" for r in rows])
    val=np.array([r["split"]=="validation" for r in rows])
    best=None
    for alpha in (1.,10.,100.,1000.):
        classifier=fit(x[train],np.eye(3)[y[train]],alpha)
        scores=predict(classifier,x[val])
        for bias in (-.25,0.,.25,.50):
            p=np.argmax(scores+np.array([bias,0.,0.]),axis=1)
            key=(macro(y[val],p),-float(np.mean(p[y[val]==0]!=0)))
            if best is None or key>best[0]:
                best=(key,alpha,bias,classifier)
    key,alpha,bias,classifier=best
    positive=train&(y!=0)
    options=[]
    for da in (1.,10.,100.,1000.):
        m=fit(x[positive],d[positive],da)
        v=val&(y!=0)
        err=np.abs(predict(m,x[v]).clip(0,HORIZON)-d[v]).mean()
        options.append((float(err),da,m))
    err,da,regressor=min(options,key=lambda v:v[0])
    return classifier,regressor,bias,dict(direction_alpha=alpha,none_score_bias=bias,distance_alpha=da,
        validation_macro_recall=key[0],validation_false_alarm_rate=-key[1],validation_distance_mae_m=err)


def metrics(rows,pred,dist):
    y=np.array([r["direction"] for r in rows])
    d=np.array([r["distance"] or 0. for r in rows])
    confusion=np.zeros((3,3),dtype=int)
    np.add.at(confusion,(y,pred),1)
    positives=y!=0
    correct=positives&(y==pred)
    groups={}
    for i,r in enumerate(rows):
        if positives[i]:
            groups.setdefault((r["run"],r["event"]),[]).append(i)
    leads=[]
    for (run,event),idx in groups.items():
        # Require two consecutive eligible captures for a stable hit.
        hits=[d[a] for a,b in zip(idx,idx[1:]) if correct[a] and correct[b]
              and rows[b]["frame"]==rows[a]["frame"]+1]
        leads.append(dict(run=run,event=event,direction=CLASSES[y[idx[0]]],n=len(idx),
                         stable_first_lead_m=float(max(hits)) if hits else None))
    return dict(n=len(rows),confusion=confusion.tolist(),false_alarms=int(((y==0)&(pred!=0)).sum()),
                negative_n=int((y==0).sum()),positive_n=int(positives.sum()),direction_hits=int(correct.sum()),
                distance_mae_m=float(np.abs(dist[positives]-d[positives]).mean()) if positives.any() else None,
                distance_mae_correct_direction_m=float(np.abs(dist[correct]-d[correct]).mean()) if correct.any() else None,
                stable_event_leads=leads)


def evaluate(out, flyvis):
    rows=[json.loads(line) for line in (out/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    pixel=np.load(out/"image-features.npy")
    excluded_history={(r["run"],i) for r in rows if r["split"]=="excluded"
                      for i in range(r["frame"]-2,r["frame"]+1)}
    clean=np.array([r["split"] not in ("train","validation") or not any(
        (r["run"],i) in excluded_history for i in range(r["frame"]-2,r["frame"]+1)) for r in rows])
    models={"linear":pixel[clean]}
    mask=clean
    if flyvis:
        with np.load(flyvis,allow_pickle=False) as f:
            check(len(f["available"])==len(rows),"FlyVis index contract")
            mask=clean&f["available"]
            models={"linear":pixel[mask],"flyvis":f["features"][mask],
                    "linear_plus_flyvis":np.column_stack((pixel[mask],f["features"][mask]))}
    rows=[r for r,m in zip(rows,mask) if m]
    results={}
    for name,x in models.items():
        x=x.astype(np.float64)
        cls,reg,bias,selection=select_model(x,rows)
        score=predict(cls,x)+np.array([bias,0.,0.])
        pred=score.argmax(1)
        distance=predict(reg,x).clip(0,HORIZON)
        vals={}
        for split in ("validation","heldout","test_other","external"):
            mask=np.array([r["split"]==split for r in rows])
            if mask.any():
                vals[split]=metrics([r for r,k in zip(rows,mask) if k],pred[mask],distance[mask])
        results[name]=dict(selection=selection,results=vals)
        keys=("mean","scale","offset","coefficients")
        np.savez_compressed(out/f"{name}-weights.npz",**dict(zip(("cls_"+k for k in keys),cls)),
                            **dict(zip(("dist_"+k for k in keys),reg)),bias=bias)
        jsonl(out/f"{name}-predictions.jsonl",(dict(r,predicted=CLASSES[c],predicted_distance_m=float(d) if c else None)
              for r,c,d in zip(rows,pred,distance)))
    counts={}
    for split in sorted({r["split"] for r in rows}):
        counts[split]=dict(Counter(CLASSES[r["direction"]] for r in rows if r["split"]==split))
    write(out/("result.json" if flyvis else "baseline-result.json"),dict(status="completed",
        flyvis_features_sha256=digest(flyvis) if flyvis else None,
        rows_sha256=digest(out/"rows.jsonl"), excluded_history_rows_removed=int((~clean).sum()),program_sha256=digest(Path(__file__)),counts=counts,
        termination="all fixed-weight evaluations exported",randomness="none",models=results))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fit-only",action="store_true")
    p.add_argument("--flyvis",type=Path)
    a=p.parse_args()
    if not a.fit_only:
        a.out.mkdir(parents=True,exist_ok=False)
        extract(a.out)
    evaluate(a.out,a.flyvis)
