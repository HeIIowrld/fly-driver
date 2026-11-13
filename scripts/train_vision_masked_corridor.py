"""Fit per-distance corridor geometry and abstain on unsupported sections."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from train_vision_road_corridor import (
    magione_corridor, monza_corridor, external_rows, old_model, path_from_corridor,
    expand,
)
from train_vision_teacher_line import (
    H, check, digest, examples, fit, groups, inputs, jsonl, predict, read, road_model, write,
)

X = np.array([0., 10., 15., 20.])
MIN_WIDTH = 1.8
MAX_WIDTH = 24.
# Approximate horizontal crop from the front mount and 56-degree nominal FOV.
HALF_FOV = np.deg2rad(39.)
CAMERA_FORWARD = 2.5
CAMERA_CONFIG = Path("configs/ac-csp-exterior-front-v3-retina.json")


def registered_horizontal_support(bounds, cameras):
    """Each edge must project within at least one registered horizontal view."""
    points = np.stack((np.broadcast_to(X[:, None], bounds.shape), bounds), axis=-1)
    covered = np.zeros(bounds.shape, dtype=bool)
    for camera in cameras.values():
        relative = points - np.asarray(camera["origin_body_m"][:2])
        depth = relative @ np.asarray(camera["forward_body"][:2])
        lateral = relative @ np.asarray(camera["right_body"][:2])
        covered |= (depth > 0.) & (np.abs(lateral) * camera["projection_x"] <= depth)
    return covered.all(axis=1)


def section_targets(row, corridor, cameras=None):
    edges = []
    for start,end,tree in corridor.edges:
        ids = tree.query_ball_point(row["xy"],42.)
        a,b = (v[ids]-row["xy"] for v in (start,end))
        a,b = a@row["rot"],b@row["rot"]
        edge = []
        for x in X:
            mask = ((a[:,0]<=x)&(b[:,0]>x)) | ((b[:,0]<=x)&(a[:,0]>x))
            lo,hi = a[mask],b[mask]
            y = lo[:,1]+(hi[:,1]-lo[:,1])*(x-lo[:,0])/(hi[:,0]-lo[:,0])
            local = y[np.abs(y)<16.]
            edge.append(float(local[0]) if len(local)==1 else np.nan)
        edges.append(edge)
    bounds = np.array(edges).T
    width = bounds[:,0]-bounds[:,1]
    geometry = np.isfinite(width)&(width>=MIN_WIDTH)&(width<=MAX_WIDTH)
    # x=0 is behind the bumper camera and is never a supervised visual target.
    if cameras is None:
        covered = np.max(np.abs(bounds),axis=1)<=np.maximum(0,X-CAMERA_FORWARD)*np.tan(HALF_FOV)
    else:
        covered = registered_horizontal_support(bounds, cameras)
    visible = geometry & (X > 0.) & covered
    return bounds,geometry,visible


def pack(row, origin, path, bounds, valid, score, turn, visibility):
    return dict(schema="flybrain.racing_line.v1", episode_id=row["run"],
        clock_id="host-qpc-recording-relative", observation_time_s=row["t"]-origin,
        ego_frame="body_xy_forward_left",origin="vehicle_reference_point",source_speed_mps=row["speed"],
        points=[dict(s_m=float(h),x_m=float(p[0]) if ok else None,
                     y_m=float(p[1]) if ok else None,valid=bool(ok))
                for h,p,ok in zip(H,path,np.isfinite(path[:,0]))],
        road_geometry=dict(visibility=visibility,
            sections=[dict(forward_x_m=float(x),left_y_m=float(b[0]) if ok else None,
                           right_y_m=float(b[1]) if ok else None,valid=bool(ok),support_score=float(s))
                      for x,b,ok,s in zip(X[1:],bounds,valid,score)],bend=turn))


def geometry_input(rows, corridor, cameras=None):
    labels = [section_targets(row,corridor,cameras) for row in rows]
    return tuple(np.stack(values) for values in zip(*labels))


def fit_sections(train,val,tc,vc):
    x,xv = inputs(train,"vision"),inputs(val,"vision")
    t,tg,tv = tc
    v,vg,vv = vc
    heads,summary = [],[]
    for i in range(1,len(X)):
        check(tv[:,i].sum()>=30 and vv[:,i].sum()>=10,"insufficient section supervision")
        width = (t[:,i,0]-t[:,i,1])[tv[:,i]]
        target = np.column_stack((t[:,i].mean(1)[tv[:,i]],np.log(width/2)))
        choices = []
        for alpha in (10.,100.,1000.):
            geo = fit(x[tv[:,i]],target,alpha)
            error = np.abs(expand(predict(geo,xv[vv[:,i]]))[:,0]-v[vv[:,i],i]).mean()
            choices.append((float(error),alpha,geo))
        error,alpha,geo = min(choices,key=lambda a:a[0])
        support = fit(x,tv[:,i].astype(float),alpha)
        raw = predict(support,xv)
        predicted = expand(predict(geo,xv))[:,0]
        reliable = vv[:,i] & (np.max(np.abs(predicted-v[:,i]),axis=1)<=2.)
        thresholds = [.35,.45,.55,.65,.75,.85,.95]
        options = []
        for threshold in thresholds:
            take = raw>=threshold
            if take.sum()>=max(10,int(.25*vv[:,i].sum())):
                options.append((float(reliable[take].mean()),int(take.sum()),threshold))
        precision,covered,threshold = max(options) if options else (None,0,1.01)
        heads.append((geo,support,threshold))
        summary.append(dict(distance=float(X[i]), train_supported=int(tv[:,i].sum()),
            validation_supported=int(vv[:,i].sum()),alpha=alpha,validation_boundary_mae_m=error,
            threshold=threshold,validation_accepted=covered,validation_supported_within_2m_fraction=precision))
    return heads,summary


def infer(heads,x):
    bounds = np.stack([expand(predict(g,x))[:,0] for g,_,_ in heads],axis=1)
    score = np.stack([predict(s,x) for _,s,_ in heads],axis=1)
    ok = score>=np.array([q for _,_,q in heads])
    return bounds,score,ok


def path_and_turn(bounds,ok):
    path = np.full((len(H),2),np.nan)
    count = int(np.cumprod(ok).sum())
    if not count:
        return path,None
    xx = X[:count+1]
    mid = bounds[:count].mean(1)
    # Grow smoothly from the body origin instead of snapping to the 5 m midpoint.
    y = np.clip(mid*(1-np.exp(-xx[1:]/9)),
                bounds[:count,1]+.6,bounds[:count,0]-.6)
    y = np.r_[0.,y]
    arc = np.r_[0.,np.cumsum(np.hypot(np.diff(xx),np.diff(y)))]
    use = H<=arc[-1]+1e-9
    path[use] = np.column_stack((np.interp(H[use],arc,xx),np.interp(H[use],arc,y)))
    if count<3:
        return path,None
    change = np.arctan2(mid[2]-mid[1],X[3]-X[2])-np.arctan2(mid[1]-mid[0],X[2]-X[1])
    return path,int(change>.10)-int(change<-.10)


def previous_turn(encoded):
    from train_vision_road_corridor import turn
    return np.array([turn(b.mean(1)) for b in expand(encoded)])


def weighted_errors(p,t,mask):
    return [float(np.abs(p[mask[:,i],i,1]-t[mask[:,i],i,1]).mean()) if mask[:,i].any()
            else None for i in range(len(H))]


def assess(name,rows,corridor,labels,heads,old,old_corridor,target,masks,out,visibility):
    x = inputs(rows,"vision")
    prediction,score,supported = infer(heads,x)
    answers = [path_and_turn(b,k) for b,k in zip(prediction,supported)]
    path = np.stack([p for p,_ in answers])
    turn = np.array([2 if y is None else y for _,y in answers])
    base = predict(old,x).reshape(-1,3,2)
    old_encoded = predict(old_corridor,x)
    prev = np.stack([path_from_corridor(b) for b in expand(old_encoded)])
    prev_turn = previous_turn(old_encoded)
    truth,geometry,visible = labels
    # Use the same original 0-to-30m bend definition for old and new comparisons.
    gt = [corridor.labels(row) for row in rows]
    from train_vision_road_corridor import turn as truth_turn
    known = np.array([b is not None for b in gt])
    bend = np.array([truth_turn(b.mean(1)) if b is not None else 2 for b in gt])
    origin = min(row["t"] for row in rows)
    jsonl(out/f"{name}-predictions.jsonl",(pack(r,origin,p,b,k,s,None if d==2 else int(d),visibility)
        for r,p,b,k,s,d in zip(rows,path,prediction,supported,score,turn)))
    result = {}
    for key,mask in masks.items():
        if not mask.any():
            continue
        emit = mask[:,None]&np.isfinite(path[:,:,0])
        full = np.broadcast_to(mask[:,None],emit.shape)
        straight = mask&known&(bend==0)
        turn_known = mask&known
        scored_turn = turn_known&(turn!=2)
        coverage = mask[:,None]&supported
        checked = coverage&visible[:,1:]
        result[key] = dict(n=int(mask.sum()),path_emit_n=emit.sum(0).tolist(),
            new_mae_on_emitted=weighted_errors(path,target,emit),
            direct_mae_on_same=weighted_errors(base,target,emit),
            old_corridor_mae_on_same=weighted_errors(prev,target,emit),
            direct_mae_full=weighted_errors(base,target,full),
            old_corridor_mae_full=weighted_errors(prev,target,full),
            boundary_emit_n=coverage.sum(0).tolist(),proxy_supported_n=(mask[:,None]&visible[:,1:]).sum(0).tolist(),
            emitted_but_proxy_unsupported_n=(coverage&~visible[:,1:]).sum(0).tolist(),
            boundary_mae_on_supported_and_emitted=[
                np.abs(prediction[checked[:,i],i]-truth[checked[:,i],i+1]).mean(0).tolist()
                if checked[:,i].any() else None for i in range(3)],
            bend_truth_n=int(turn_known.sum()), bend_emit_n=int(scored_turn.sum()),
            old_bend_correct=int((turn_known&(prev_turn==bend)).sum()),
            new_bend_correct=int((scored_turn&(turn==bend)).sum()),
            old_bend_correct_on_same=int((scored_turn&(prev_turn==bend)).sum()),
            straight_truth_n=int(straight.sum()),straight_new_emit_n=int((straight&(turn!=2)).sum()),
            old_straight_false_turn=int((straight&(prev_turn!=0)).sum()),
            old_straight_false_turn_on_same=int((straight&(turn!=2)&(prev_turn!=0)).sum()),
            new_straight_false_turn=int((straight&(turn!=2)&(turn!=0)).sum()))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--source-revision",required=True)
    ap.add_argument("--registered-views",action="store_true")
    args=ap.parse_args()
    cv2.setNumThreads(1)
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    cameras = read(CAMERA_CONFIG)["views"] if args.registered_views else None
    visibility = ("estimated_registered_horizontal_support" if cameras is not None
                  else "estimated_front_horizontal_support")
    b=Path("runs/braking-corner-learning-20261105")
    track=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
    direct_file=Path("runs/vision-teacher-line-20261110-02/vision-weights.npz")
    previous_file=Path("runs/vision-road-corridor-20261110-02/corridor-weights.npz")
    direct=old_model(direct_file)
    with np.load(previous_file,allow_pickle=False) as z:
        previous=tuple(z[k] for k in ("mean","scale","offset","coefficients"))
    road=road_model(track)
    corridor=monza_corridor(track)
    d=b/"monza-user-started-teacher-01"
    develop,dp=examples(d,d/"qualification-03",road,True,out)
    train=[r for r in develop if r["split"]=="train"]
    val=[r for r in develop if r["split"]=="validation"]
    heads,details=fit_sections(train,val,geometry_input(train,corridor,cameras),
                               geometry_input(val,corridor,cameras))
    keys=("mean","scale","offset","coefficients")
    weights=out/"masked-corridor-weights.npz"
    np.savez_compressed(weights,**{f"{i}_{name}_{k}":v for i,(g,s,_) in enumerate(heads)
        for name,head in (("geometry",g),("support",s)) for k,v in zip(keys,head)},
        thresholds=np.array([h[2] for h in heads]))
    frozen=digest(weights)
    with np.load(weights,allow_pickle=False) as z:
        heads=[(tuple(z[f"{i}_geometry_{k}"] for k in keys),tuple(z[f"{i}_support_{k}"] for k in keys),q)
               for i,q in enumerate(z["thresholds"])]
    t=b/"fresh-monza-develop-01"
    test,tp=examples(t,t/"qualification-02",road,False,out)
    monza=assess("monza",test,corridor,geometry_input(test,corridor,cameras),heads,direct,previous,
                 np.stack([r["local"] for r in test]),groups(test,road),out,visibility)
    geo=b/"heldout-magione-geometry-01/geometry.json"
    mc=magione_corridor(geo)
    external={}
    for episode,sub in [("magione-student-parent-02","run"),("magione-student-candidate-01","run"),
                        ("magione-user-recovered-01","candidate"),("magione-alternative-student-parent-01","run")]:
        rows,prov=external_rows(b/episode/sub/"vision")
        labeled=[(r,mc.labels(r)) for r in rows]
        usable=[r for r,g in labeled if g is not None]
        truth=np.stack([path_from_corridor(g) for _,g in labeled if g is not None])
        external[episode]=dict(provenance=prov,results=assess(episode,usable,mc,geometry_input(usable,mc,cameras),
            heads,direct,previous,truth,{"all":np.ones(len(usable),dtype=bool)},out,visibility))
    check(digest(weights)==frozen,"external evaluation changed weights")
    write(out/"result.json",dict(status="completed",termination="all fixed-weight comparisons exported",
        source_revision=args.source_revision,script_sha256=digest(Path(__file__)),seed=None,visibility=visibility,
        camera_config_sha256=digest(CAMERA_CONFIG) if cameras is not None else None,
        weights_sha256=frozen,inputs_sha256={str(p):digest(p) for p in (track,geo,direct_file,previous_file,
            Path(__file__).with_name("train_vision_road_corridor.py"),Path(__file__).with_name("train_vision_teacher_line.py"))},
        train=len(train),validation=len(val),section_calibration=details,develop=dp,test=tp,monza=monza,magione=external))


if __name__=="__main__":
    main()
