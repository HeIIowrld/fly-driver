"""Offline corridor-first vision probe and frozen transfer evaluation."""
import argparse
from collections import Counter
import math
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from train_vision_teacher_line import (
    CameraArchive, H, VIEWS, check, digest, examples, export_record,
    fit, groups, image_features, inputs, jsonl, predict, read, road_model, write,
)

X = np.array([0., 5., 10., 20., 30.])
TURN_THRESHOLD = .025


class Corridor:
    def __init__(self, left, right):
        self.edges = [self.segments(edge) for edge in (left, right)]
        self.polygons = sorted([left.astype(np.float32), right.astype(np.float32)],
                               key=cv2.contourArea, reverse=True)

    @staticmethod
    def segments(edge):
        end = np.roll(edge, -1, axis=0)
        return edge, end, cKDTree((edge+end)/2)

    def labels(self, row):
        boundaries = []
        for start, end, tree in self.edges:
            ids = tree.query_ball_point(row["xy"], 48.)
            a = (start[ids]-row["xy"])@row["rot"]
            b = (end[ids]-row["xy"])@row["rot"]
            ys = []
            for x in X:
                mask = ((a[:,0]<=x)&(b[:,0]>x)) | ((b[:,0]<=x)&(a[:,0]>x))
                a0,b0 = a[mask],b[mask]
                y = a0[:,1]+(b0[:,1]-a0[:,1])*(x-a0[:,0])/(b0[:,0]-a0[:,0])
                if not len(y):
                    return None
                ys.append(y[np.argmin(np.abs(y))])
            boundaries.append(ys)
        left,right = np.array(boundaries)
        if not np.all((left-right>=1.8)&(left-right<=24)):
            return None
        return np.stack((left,right),axis=-1)


def monza_corridor(path):
    data = np.genfromtxt(path,names=True,delimiter=",")
    return Corridor(*[np.column_stack([data[f"{side}_border_{axis}"] for axis in "xy"])
                      for side in ("left","right")])


def magione_corridor(path):
    receipt = read(path.parent/"result.json")
    geometry = read(path)
    check(receipt["geometry_sha256"]==digest(path) and geometry["track_id"]=="magione",
          "Magione geometry receipt mismatch")
    rows = geometry["rows"]
    center = np.array([r["native_xyz"] for r in rows])[:,[2,0]]
    tangent = np.roll(center,-1,axis=0)-np.roll(center,1,axis=0)
    tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
    normal = np.column_stack((-tangent[:,1],tangent[:,0]))
    left,right = (np.array([r[f"{side}_m"] for r in rows]) for side in ("left","right"))
    return Corridor(center+left[:,None]*normal,center-right[:,None]*normal)


def compress(bounds):
    return np.r_[bounds.mean(axis=1),np.log((bounds[:,0]-bounds[:,1])/2)]


def expand(encoded):
    midpoint,log_width = np.split(encoded,2,axis=-1)
    half_width = np.exp(np.clip(log_width,np.log(.9),np.log(12.)))
    return np.stack((midpoint+half_width,midpoint-half_width),axis=-1)


def turn(line):
    heading_near = math.atan2(line[1]-line[0],X[1]-X[0])
    heading_far = math.atan2(line[-1]-line[-2],X[-1]-X[-2])
    change = heading_far-heading_near
    return int(change>TURN_THRESHOLD)-int(change<-TURN_THRESHOLD)


def path_from_corridor(bounds):
    midpoint = bounds.mean(axis=1)
    # Start at the body origin and converge toward the perceived corridor center.
    # Reserve at most 0.6 m on each side of the inferred center-point path.
    y = midpoint-midpoint[0]*np.exp(-X/7.)
    y = np.clip(y,bounds[:,1]+.6,bounds[:,0]-.6)
    y[0] = 0.
    arc = np.r_[0.,np.cumsum(np.hypot(np.diff(X),np.diff(y)))]
    return np.column_stack((np.interp(H,arc,X),np.interp(H,arc,y)))


def old_model(path):
    with np.load(path,allow_pickle=False) as z:
        return tuple(z[f"path_{key}"] for key in ("mean","scale","offset","coefficients"))


def external_rows(directory):
    rows = []
    counts = Counter()
    with CameraArchive(directory) as archive:
        stream = [json.loads(line) for line in archive.read("stream.jsonl").splitlines()]
        valid = {}
        for item in stream:
            if item["ideal_line_shown"] or item["aid_probe_only"]:
                counts["aid"] += 1
                continue
            i = item["capture_id"]
            meta = archive.json(Path(item["file"]).name)
            check(meta["track_id"]=="magione" and meta["index"]==i,"external camera identity")
            views = {v["view_id"]:v for v in item["views"]}
            if set(views)!=set(VIEWS):
                counts["missing_view"] += 1
                continue
            latest = max(views.values(),key=lambda v:v["render_started_lower"])
            body = latest["source_render_metadata"]["source_body_after_render"]
            if not body["physics_available"] or body["speed_m_s"] < 2.:
                counts["stationary_or_missing"] += 1
                continue
            look = np.asarray(body["renderer_only_look"])[[2,0]]
            look /= np.linalg.norm(look)
            valid[i] = dict(index=i, run=str(directory.parent.parent.name),t=item["received"],
                xy=np.asarray(body["renderer_only_position"])[[2,0]],rot=np.column_stack((look,[-look[1],look[0]])),
                speed=body["speed_m_s"],views=views,generation=item["generation"])
        features = {}
        for i,row in valid.items():
            ids = (i-2,i-1,i)
            if any(j not in valid or valid[j]["generation"]!=row["generation"] for j in ids):
                counts["missing_history"] += 1
                continue
            if not all(.045 < valid[b]["t"]-valid[a]["t"] < .18 for a,b in zip(ids,ids[1:])):
                counts["cadence"] += 1
                continue
            for j in ids:
                if j not in features:
                    pixels = [cv2.imdecode(np.frombuffer(archive.read(Path(valid[j]["views"][v]["file"]).name),
                                                np.uint8),cv2.IMREAD_GRAYSCALE) for v in VIEWS]
                    check(all(p is not None for p in pixels),"external image decode")
                    features[j] = np.concatenate([image_features(p) for p in pixels])
            rows.append(dict(row,feature=np.concatenate([features[j] for j in ids])))
        return rows,dict(bundle_sha256=archive.bundle_sha256,drops=dict(counts),frames=len(stream))


def attach_labels(rows,corridor):
    labeled = []
    for row in rows:
        bounds = corridor.labels(row)
        if bounds is not None:
            labeled.append(dict(row,bounds=bounds))
    return labeled


def metrics(rows, predictions, corridor, target):
    path = np.asarray(predictions)
    err = np.abs(path-np.asarray(target))
    outer,inner = corridor.polygons
    outside = []
    for row,pred in zip(rows,path):
        world = pred@row["rot"].T+row["xy"]
        outside.append([cv2.pointPolygonTest(outer,tuple(v),False)<0 or
                        cv2.pointPolygonTest(inner,tuple(v),False)>0 for v in world])
    return dict(n=len(rows),lateral_mae_m=err[:,:,1].mean(0).tolist(),
                point_outside_count=np.sum(outside,axis=0).tolist())


def assess(name, rows, corridor, model, baseline, baseline_speed, target, masks, output):
    x = inputs(rows,"vision")
    pred_bounds = expand(predict(model,x))
    paths = np.stack([path_from_corridor(b) for b in pred_bounds])
    old = predict(baseline,x).reshape(-1,3,2)
    old_speed = predict(baseline_speed,inputs(rows,"vision_speed")).reshape(-1,3,2)
    known = np.array([r.get("bounds") is not None for r in rows])
    truth = np.stack([r["bounds"] if ok else np.full((len(X),2),np.nan) for r,ok in zip(rows,known)])
    actual_turn = np.array([turn(b.mean(axis=1)) if ok else 0 for b,ok in zip(truth,known)])
    predicted_turn = np.array([turn(b.mean(axis=1)) for b in pred_bounds])
    results = {}
    for label, mask in masks.items():
        if not mask.any():
            continue
        subset = [r for r,take in zip(rows,mask) if take]
        labeled = mask & known
        gap = np.abs(pred_bounds[labeled]-truth[labeled])
        confusion = np.zeros((3,3),dtype=int)
        np.add.at(confusion,(actual_turn[labeled]+1,predicted_turn[labeled]+1),1)
        results[label] = dict(boundary_n=int(labeled.sum()), boundary_mae_m=gap.mean(0).tolist() if labeled.any() else None,
            bend_confusion_right_straight_left=confusion.tolist(),
            bend_accuracy=float((actual_turn[labeled]==predicted_turn[labeled]).mean()) if labeled.any() else None,
            path={k:metrics(subset,p[mask],corridor,target[mask]) for k,p in
                  (("corridor_first",paths),("old_vision",old),("old_vision_speed",old_speed),
                   ("straight",np.broadcast_to(np.c_[H,np.zeros(3)],paths.shape)))})
    t0 = min(r["t"] for r in rows)
    jsonl(output/f"{name}-predictions.jsonl",(
        dict(export_record(row,t0,path),capture_id=row["index"],observation_host_s=row["t"],
             road_geometry=dict(forward_x_m=X.tolist(),left_y_m=bound[:,0].tolist(),
                                right_y_m=bound[:,1].tolist(),bend=int(direction)))
        for row,path,bound,direction in zip(rows,paths,pred_bounds,predicted_turn)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--source-revision",required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    b = Path("runs/braking-corner-learning-20261105")
    track = Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
    prior = Path("runs/vision-teacher-line-20261110-02")
    develop,test = (b/n for n in ("monza-user-started-teacher-01","fresh-monza-develop-01"))
    road = road_model(track)
    corridor = monza_corridor(track)
    args.out.mkdir(parents=True,exist_ok=False)
    d,dp = examples(develop,develop/"qualification-03",road,True,args.out)
    train = attach_labels([r for r in d if r["split"]=="train"],corridor)
    val = attach_labels([r for r in d if r["split"]=="validation"],corridor)
    x,xv = inputs(train,"vision"),inputs(val,"vision")
    y = np.stack([compress(r["bounds"]) for r in train])
    yv = np.stack([r["bounds"] for r in val])
    candidates = []
    for alpha in (1.,10.,100.,1000.):
        model = fit(x,y,alpha)
        score = float(np.abs(expand(predict(model,xv))-yv).mean())
        candidates.append((score,alpha,model))
    loss,alpha,model = min(candidates,key=lambda v:v[0])
    weight = args.out/"corridor-weights.npz"
    np.savez_compressed(weight,**dict(zip(("mean","scale","offset","coefficients"),model)))
    frozen = digest(weight)
    with np.load(weight,allow_pickle=False) as z:
        model = tuple(z[k] for k in ("mean","scale","offset","coefficients"))
    old_path,old_speed_path = (prior/f"{n}-weights.npz" for n in ("vision","vision_speed"))
    baseline,baseline_speed = old_model(old_path),old_model(old_speed_path)
    t,tp = examples(test,test/"qualification-02",road,False,args.out)
    t = [dict(row,bounds=corridor.labels(row)) for row in t]
    monza = assess("monza",t,corridor,model,baseline,baseline_speed,
                   np.stack([r["local"] for r in t]),groups(t,road),args.out)
    geo = b/"heldout-magione-geometry-01/geometry.json"
    external = magione_corridor(geo)
    magione = {}
    for episode,sub in [("magione-student-parent-02","run"),("magione-student-candidate-01","run"),
                        ("magione-user-recovered-01","candidate"),("magione-alternative-student-parent-01","run")]:
        rows,p = external_rows(b/episode/sub/"vision")
        usable = attach_labels(rows,external)
        if usable:
            geom_target = np.stack([path_from_corridor(r["bounds"]) for r in usable])
            selection = np.ones(len(usable),dtype=bool)
            bins = np.unique(np.floor(np.stack([r["xy"] for r in usable])/5),axis=0)
            magione[episode] = dict(provenance=p, moving_rows=len(rows), occupied_xy_5m_cells=len(bins),
                mean_speed_kmh=float(np.mean([r["speed"] for r in usable])*3.6),
                result=assess(episode,usable,external,model,baseline,baseline_speed,geom_target,{"all":selection},args.out))
        else:
            magione[episode] = dict(provenance=p,moving_rows=len(rows),status="no_geometry_overlap")
    check(digest(weight)==frozen,"frozen weights changed")
    write(args.out/"result.json",dict(status="completed",termination="all archived observations evaluated",
        source_revision=args.source_revision,source_sha256=digest(Path(__file__)),randomness="none",
        weights_sha256=frozen,old_vision_sha256=digest(old_path),old_vision_speed_sha256=digest(old_speed_path),
        helper_source_sha256=digest(Path(__file__).with_name("train_vision_teacher_line.py")),
        monza_map_sha256=digest(track),magione_map_sha256=digest(geo),train=len(train),validation=len(val),
        alpha=alpha,validation_boundary_mae_m=loss,x_sections_m=X.tolist(),bend_threshold_rad=TURN_THRESHOLD,
        develop=dp,test=tp,monza=monza,magione=magione))


if __name__=="__main__":
    main()
