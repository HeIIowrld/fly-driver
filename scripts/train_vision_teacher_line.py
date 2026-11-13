"""Train an offline vision-only and vision+speed teacher-line probe."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.linalg import solve

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from verify_ac_heldout_curvature import road_model

VIEWS = ("front", "left", "right", "rear")
H = np.array([5., 10., 20.])
HOLD = 120.
CORNER = .008
VALIDATION = [(300., 600.), (2820., 3220.), (5100., 5450.)]
NAMES = ("none", "left", "right")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_teacher(run, qualification, *, recording=None):
    recording = run / 'demonstration' if recording is None else Path(recording)
    proof_file = qualification / "lap-proof/result.json"
    qual = read(qualification / "result.json")
    proof = read(proof_file)
    check(digest(proof_file) == qual["lap_proof_sha256"], "lap proof hash")
    check(proof["verified_valid_laps"] == 1 and proof["laps"][0]["zero_tyres_out"], "expected clean lap")
    tcp = recording / "teacher/tcp-source-packets.jsonl"
    tcp_hash = digest(tcp)
    check(tcp_hash == read(tcp.parent / "result.json")["tcp_source_archive_sha256"], "TCP archive hash")
    start, end = (x["source_time"] for x in proof["crossings"])
    rows = []
    with tcp.open(encoding="utf-8") as stream:
        for line in stream:
            receipt = json.loads(line)
            wire = base64.b64decode(receipt["wire_base64"])
            packet = json.loads(wire)
            if not start <= packet["timestamp_ac"] <= end:
                continue
            check(hashlib.sha256(wire).hexdigest() == receipt["wire_sha256"], "TCP packet hash")
            raw = packet["privileged_teacher"]["raw"]
            snap = packet["source_native_snapshot"]
            native = PHYSICS_PREFIX.unpack(base64.b64decode(snap["prefix_base64"]))
            hz = snap["qpc_frequency_hz"]
            check(snap["repeated_copy_equal"] and hz == 10_000_000 and raw["LapInvalidated"] == 0, "packet eligibility")
            rows.append((snap["qpc_started_ticks"]/hz, snap["qpc_finished_ticks"]/hz,
                         raw["world_position_x"], raw["world_position_y"], native[7]/3.6))
    data = np.array(rows)
    check(len(data) == proof["laps"][0]["sdk_source_packets"]+1, "lap count")
    check(np.all(np.diff(data[:, :2], axis=0) > 0), "clock order")
    xy = data[:, 2:4]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    check(np.all(np.diff(arc)>0), "path must advance")
    return data, arc, qual, dict(tcp_sha256=tcp_hash, lap_proof_sha256=digest(proof_file),
                                 official_lap_s=proof["laps"][0]["lap_time_ms"]/1000, origin_host_qpc_s=float(data[0,0]))


def at(arc, axis, query):
    return np.column_stack([np.interp(np.atleast_1d(query), arc, column) for column in axis.T])


def corners(arc, xy):
    grid = np.arange(0., arc[-1], 1.)
    p = gaussian_filter1d(at(arc, xy, grid), 3., axis=0)
    d = np.gradient(p, axis=0)
    dd = np.gradient(d, axis=0)
    k = (d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.maximum(.01,np.linalg.norm(d,axis=1))**3
    sign = np.where(k >= CORNER, 1, np.where(k <= -CORNER, -1, 0))
    spans = []
    start = 0
    for j in range(1, len(sign)+1):
        if j == len(sign) or sign[j] != sign[start]:
            if sign[start] and j-start >= 8:
                spans.append((int(start), int(j), int(sign[start])))
            start = j
    return spans


def image_features(gray):
    pixels = cv2.resize(gray, (24, 14), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    gx = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=1)
    coarse = cv2.resize(pixels, (12,7), interpolation=cv2.INTER_AREA)
    bins = np.floor((np.arctan2(gy,gx)%np.pi)*(6/np.pi)).astype(int).clip(0,5)
    magnitude = np.hypot(gx,gy)
    hist = []
    for y in range(0,14, 4):
        for x in range(0,24, 4):
            tile = np.s_[y:y+4,x:x+4]
            values = np.bincount(bins[tile].ravel(), magnitude[tile].ravel(), minlength=6)
            hist.extend(values/(.10+values.sum()))
    return np.r_[coarse.ravel(),hist].astype(np.float32)


def points(values):
    return [dict(s_m=float(h), x_m=float(x), y_m=float(y), valid=True) for h,(x,y) in zip(H,values)]


def edge_polygons(track):
    d = np.genfromtxt(track, names=True, delimiter=",")
    a,b = (np.column_stack([d[f"{side}_border_{axis}"] for axis in "xy"]).astype(np.float32)
           for side in ("left","right"))
    return sorted([a,b], key=cv2.contourArea, reverse=True)


def examples(run, qualification, road, train, output, *, recording=None):
    recording = run / 'demonstration' if recording is None else Path(recording)
    data, arc, qual, provenance = load_teacher(run, qualification, recording=recording)
    pairing_file = qualification/"pairings.json"
    check(digest(pairing_file) == qual["pairings_sha256"], "camera pairings hash")
    allowed = {r["camera_index"] for r in read(pairing_file) if r["eligible"]}
    spans = corners(arc,data[:,2:4])
    _, grid, tree, length, holdouts = road
    path = recording/"camera/vision"
    drops = Counter()
    rows = []
    with CameraArchive(path) as archive:
        check(archive.bundle_sha256 == qual["camera_bundle_sha256"], "camera qualification hash")
        clock = archive.json("clock-initial.json")
        quant = clock.get("host_clock_quantization_seconds",0.)
        low = clock["offset_lower_seconds"]-quant
        high = clock["offset_upper_seconds"]+quant
        stream = [json.loads(x) for x in archive.read("stream.jsonl").splitlines()]
        frames = {}
        for i, item in enumerate(stream):
            if i not in allowed or item["ideal_line_shown"] or item["aid_probe_only"]:
                drops["camera_qualification"] += 1
                continue
            meta = archive.json(Path(item["file"]).name)
            check(i == item["capture_id"] == meta["index"] and meta["generation"] == clock["generation"], "camera identity")
            views = {v["view_id"]:v for v in item["views"]}
            check(set(views) == set(VIEWS), "camera views")
            original_views = {v["view_id"]: v for v in meta["views"]}
            check(set(original_views) == set(VIEWS), "original camera views")
            raw = [views[v].get("source_render_metadata", original_views[v]) for v in VIEWS]
            lower = np.array([x["render_started_clock"]+low for x in raw])
            upper = np.array([x["render_call_finished_clock"]+high for x in raw])
            check(max(abs(t-views[v]["render_started_lower"]) for t,v in zip(lower,VIEWS)) < .001, "render clock mismatch")
            anchor = float(max(upper))
            if min(lower) < data[0,0] or anchor >= data[-1,0]:
                drops["outside_lap"] += 1
                continue
            pos = at(data[:,0],data[:,2:4],anchor)[0]
            front = raw[0]["source_body_before_render"]
            witness = np.array(front["renderer_only_position"])[[2,0]]
            front_time = raw[0]["render_started_clock"]+(low+high)/2
            front_position = at(data[:,0],data[:,2:4],front_time)[0]
            error = float(np.linalg.norm(witness-front_position))
            if error > 2.:
                drops["front_pose_residual_over_2m"] += 1
                continue
            look = np.array(raw[int(np.argmax(upper))]["source_body_after_render"]["renderer_only_look"])[[2,0]]
            look /= np.linalg.norm(look)
            side = np.array([-look[1],look[0]])
            station = float(np.interp(anchor,data[:,0],arc))
            distance, ix = tree.query(pos)
            if distance > 7:
                drops["map_registration"] += 1
                continue
            if station+H[-1] > arc[-1]:
                drops["insufficient_20m_followup"] += 1
                continue
            goal = at(arc,data[:,2:4],station+H)
            rotation = np.column_stack((look,side))
            local = (goal-pos)@rotation
            check(np.all(local[:,0] > 0) and np.allclose(local@rotation.T+pos,goal), "body frame label")
            speed_index = np.searchsorted(data[:,1], anchor, side="right")-1
            age = float(anchor-data[speed_index,1])
            if age > .035:
                drops["causal_speed_gap"] += 1
                continue
            nxt = next((c for c in spans if c[0]>=station and c[0]-station<=HOLD),None)
            current_span = next((c for c in spans if c[0]<=station<c[1]),None)
            current = current_span[2] if current_span else 0
            frames[i] = dict(t=anchor, min_render=float(min(lower)), render_span_s=float(max(upper)-min(lower)),
                             residual=error, views=views, generation=meta["generation"],
                             local=local, speed=float(data[speed_index,4]), speed_age_s=age,
                             station=station, xy=pos, rot=rotation, map_s=float(grid[ix]),
                             current=current, corner_start=(current_span[0] if current_span else None), upcoming=(0 if nxt is None else (1 if nxt[2]>0 else 2)),
                             next_valid=bool(station+HOLD<=arc[-1]), next_distance=(None if nxt is None else float(nxt[0]-station)))
        feature_cache = {}

        def feature(i):
            if i not in feature_cache:
                gray = [cv2.imdecode(np.frombuffer(archive.read(Path(frames[i]["views"][v]["file"]).name),np.uint8),
                                     cv2.IMREAD_GRAYSCALE) for v in VIEWS]
                check(all(x is not None for x in gray), "image decode")
                feature_cache[i] = np.concatenate([image_features(x) for x in gray])
            return feature_cache[i]

        def mask_of(map_s):
            gap = np.abs((map_s-holdouts+length/2)%length-length/2)
            if min(gap)<=90:
                return "excluded"
            return "validation" if any(lo<=map_s<=hi for lo,hi in VALIDATION) else "train"

        for i,f in frames.items():
            past = [i-2,i-1,i]
            if any(j not in frames or frames[j]["generation"]!=f["generation"] for j in past):
                drops["missing_history"] += 1
                continue
            ts = np.array([frames[j]["t"] for j in past])
            if not np.all((np.diff(ts)>.045)&(np.diff(ts)<.18)):
                drops["history_cadence"] += 1
                continue
            if train:
                path_start = float(np.interp(frames[i-2]["min_render"],data[:,0],arc))
                guard_end = min(f["station"]+HOLD,arc[-1])
                probe_s = np.r_[np.arange(path_start,guard_end,1.), guard_end]
                sample_map = grid[tree.query(at(arc,data[:,2:4],probe_s))[1]]
                split = mask_of(f["map_s"])
                if split=="excluded" or any(mask_of(x)!=split for x in sample_map):
                    drops["split_guard"] += 1
                    split = "excluded"
            else:
                split = "test"
            rows.append(dict(f, index=i, split=split, run=run.name, history=past, history_t=ts.tolist(),
                             feature=(np.concatenate([feature(j) for j in past]) if split!="excluded" else None)))
        if train:
            chosen = []
            for label, sign in [("straight",0),("left",1),("right",-1)]:
                options = [r for r in rows if r["current"]==sign and
                           (abs(r["local"][-1,1]) < .15 if sign==0 else sign*r["local"][-1,1] > 1)]
                check(bool(options), f"no {label} audit example")
                row = sorted(options,key=lambda r:r["residual"])[0]
                front = archive.read(Path(row["views"]["front"]["file"]).name)
                img = cv2.imdecode(np.frombuffer(front,np.uint8),cv2.IMREAD_COLOR)
                canvas = np.full((180, 320, 3), 245, dtype=np.uint8)
                def px(x,y):
                    return int(160-y*16),int(158-x*6)
                cv2.line(canvas,px(0,0),px(21,0),(155,155,155),1)
                prev=px(0,0)
                for x,y in row["local"]:
                    pos=px(x,y)
                    cv2.line(canvas,prev,pos,(60,110,215),2)
                    cv2.circle(canvas,pos,4,(40,40,170),-1)
                    prev=pos
                cv2.putText(canvas,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
                cv2.imwrite(str(output/f"audit-{label}.png"),np.hstack([cv2.resize(img,(320,180)),canvas]))
                chosen.append(dict(kind=label,capture_id=row["index"],front_pose_residual_m=row["residual"],
                                   map_arc_m=row["map_s"],points=points(row["local"])))
            write(output/"label-audit.json", chosen)
    check(bool(rows),"no usable rows")
    provenance.update(camera_bundle_sha256=qual["camera_bundle_sha256"],pairings_sha256=digest(pairing_file),
                      drops=dict(drops), corner_spans_m=spans, retained=len(rows), render_uncertainty_s=high-low)
    return rows, provenance

def fit(x, y, alpha):
    mu, sd = x.mean(axis=0), x.std(axis=0).clip(.04)
    z = (x-mu)/sd
    ym = y.mean(axis=0)
    if z.shape[0]<z.shape[1]:
        gram = z@z.T
        gram.flat[::len(gram)+1] += alpha
        beta = z.T@solve(gram,y-ym,assume_a="pos")
    else:
        gram = z.T@z
        gram.flat[::len(gram)+1] += alpha
        beta = solve(gram,z.T@(y-ym),assume_a="pos")
    return mu,sd,ym,beta


def predict(model,x):
    mu,sd,ym,beta = model
    return ym + ((x-mu)/sd)@beta


def inputs(rows,name):
    speed = np.array([r["speed"] for r in rows])[:,None]/50
    if name=="speed_only":
        return speed
    x = np.stack([r["feature"] for r in rows]).astype(np.float64)
    return np.column_stack((x,speed)) if name=="vision_speed" else x


def export_record(r, origin, path):
    return dict(schema="flybrain.racing_line.v1",episode_id=r["run"],
                clock_id="host-qpc-lap-relative", observation_time_s=r["t"]-origin,
                ego_frame="body_xy_forward_left",origin="vehicle_reference_point",
                source_speed_mps=r["speed"],points=points(path))


def metrics(rows, path, score, dist, borders):
    truth = np.stack([r["local"] for r in rows])
    err = np.abs(path-truth)
    outer,inner = borders
    def violation(r, vals):
        world = vals@r["rot"].T+r["xy"]
        return [cv2.pointPolygonTest(outer,tuple(v),False)<0 or
                cv2.pointPolygonTest(inner,tuple(v),False)>0 for v in world]
    pred_out=np.array([violation(r,v) for r,v in zip(rows,path)])
    teacher_out=np.array([violation(r,v) for r,v in zip(rows,truth)])
    target = np.array([r["upcoming"] for r in rows])
    available = np.array([r["next_valid"] for r in rows])
    pred = np.argmax(score,axis=1)
    confusion=np.zeros((3,3),dtype=int)
    np.add.at(confusion,(target[available],pred[available]),1)
    event = available & (target!=0)
    true_distance=np.array([r["next_distance"] or 0. for r in rows])
    return dict(n=len(rows), lateral_mae_m=err[:,:,1].mean(0).tolist(),
                lateral_p95_m=np.percentile(err[:,:,1],95,axis=0).tolist(),
                forward_mae_m=err[:,:,0].mean(0).tolist(),
                point_boundary_violation_count=pred_out.sum(0).tolist(),
                teacher_point_boundary_violation_count=teacher_out.sum(0).tolist(),
                point_boundary_violation_rate=pred_out.mean(0).tolist(),
                next_n=int(available.sum()), next_accuracy=float(np.mean(pred[available]==target[available])) if available.any() else None,
                next_confusion_rows_truth=confusion.tolist(),
                next_events=int(event.sum()),
                next_advance_bins={f"{lo}-{hi}m":dict(n=int(m.sum()), accuracy=float(np.mean(pred[m]==target[m])) if m.any() else None)
                    for lo,hi in [(0,20),(20,60),(60,121)]
                    for m in [event & (true_distance>=lo) & (true_distance<hi)]}, next_distance_mae_m=float(np.mean(np.abs(dist[event]-true_distance[event]))) if event.any() else None)


def groups(rows, road):
    length,holdouts=road[-2:]
    masks = {"lap":np.ones(len(rows),dtype=bool)}
    current = np.array([r["current"] for r in rows])
    for name, sign in [("straight",0),("left",1),("right",-1)]:
        masks[name]=current==sign
    m = np.array([r["map_s"] for r in rows])
    for c in holdouts:
        masks[f"heldout_map_{int(c)}"] = np.abs((m-c+length/2)%length-length/2)<=32
    corner = np.array([r["corner_start"] if r["corner_start"] is not None else -1 for r in rows])
    for c in np.unique(corner[corner>=0]):
        masks[f"corner_at_teacher_arc_{c}m"] = corner==c
    return {k:v for k,v in masks.items() if v.any()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--develop",type=Path,required=True)
    parser.add_argument("--develop-qualification",default="qualification-03")
    parser.add_argument("--test",type=Path,required=True)
    parser.add_argument("--test-qualification",default="qualification-02")
    parser.add_argument("--track",type=Path,required=True)
    parser.add_argument("--external",action="append",nargs=3,default=[],
                        metavar=("RUN","QUALIFICATION","TRACK_CSV"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--source-revision",required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    check(args.develop.resolve()!=args.test.resolve(),"distinct laps required")
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    road=road_model(args.track)
    develop, dp=examples(args.develop,args.develop/args.develop_qualification,road,True,out)
    train=[r for r in develop if r["split"]=="train"]
    val=[r for r in develop if r["split"]=="validation"]
    truth=np.stack([r["local"] for r in train])
    validation=np.stack([r["local"] for r in val])
    next_valid=np.array([r["next_valid"] for r in train])
    categories=np.array([r["upcoming"] for r in train])
    event=next_valid & (categories!=0)
    d=np.array([r["next_distance"] or 0. for r in train])
    params={}
    for name in ("speed_only","vision","vision_speed"):
        x=inputs(train,name)
        xv=inputs(val,name)
        cand=[]
        for alpha in (1.,10.,100.,1000.):
            model=fit(x,truth.reshape(len(train),6),alpha)
            loss=np.abs(predict(model,xv).reshape(-1,3,2)[:,:,1]-validation[:,:,1]).mean()
            cand.append((float(loss),alpha,model))
        loss,alpha,model=min(cand,key=lambda v:v[0])
        classifier=fit(x[next_valid],np.eye(3)[categories[next_valid]],alpha)
        distance=fit(x[event],d[event],alpha)
        params[name]=(model,classifier,distance,alpha,loss)
        np.savez_compressed(out/f"{name}-weights.npz",**{f"{head}_{key}":v
            for head, fitted in zip(("path","direction","distance"),(model,classifier,distance))
            for key,v in zip(("mean","scale","offset","coefficients"),fitted)})
    test,tp=examples(args.test,args.test/args.test_qualification,road,False,out)
    reports=[("monza_test",test,tp,road,edge_polygons(args.track),args.track)]
    for run, qualification, track in args.external:
        run,track=Path(run),Path(track)
        check(run.resolve() not in (args.develop.resolve(),args.test.resolve()),"external lap must be distinct")
        ext_road=road_model(track)
        rr,rp=examples(run,run/qualification,ext_road,False,out)
        reports.append((run.name,rr,rp,ext_road,edge_polygons(track),track))
    all_rows=develop+[r for _,rows,*_ in reports for r in rows]
    origins={args.develop.name:dp["origin_host_qpc_s"]}
    origins.update({rows[0]["run"]:prov["origin_host_qpc_s"] for _,rows,prov,*_ in reports})
    jsonl(out/"teacher-labels.jsonl", (dict(export_record(r,origins[r["run"]],r["local"]),
          capture_id=r["index"],split=r["split"],host_qpc_s=r["t"], history_capture_ids=r["history"],
          history_host_qpc_s=r["history_t"], render_span_s=r["render_span_s"], front_pose_residual_m=r["residual"],
          speed_age_s=r["speed_age_s"], map_arc_m=r["map_s"], current_turn=r["current"],
          next_corner=dict(valid=r["next_valid"],direction=NAMES[r["upcoming"]],distance_m=r["next_distance"]))
          for r in all_rows))
    results={}
    for label, rows, provenance, rd, borders, track in reports:
        results[label]=dict(provenance=provenance, track_sha256=digest(track), models={})
        masks=groups(rows,rd)
        n=len(rows)
        for name in ("straight","mean",*params):
            if name in params:
                model,cls,dist,_,_=params[name]
                x=inputs(rows,name)
                pred=predict(model,x).reshape(n,3,2)
                score=predict(cls,x)
                distance=predict(dist,x).clip(0,HOLD)
            else:
                line=np.c_[H,np.zeros(3)] if name=="straight" else truth.mean(0)
                pred=np.repeat(line[None],n,axis=0)
                score=np.repeat(np.eye(3)[categories[next_valid]].mean(0)[None],n,axis=0)
                distance=np.repeat(np.median(d[event]),n)
            results[label]["models"][name]={k:metrics([r for r,v in zip(rows,mask) if v],
                pred[mask],score[mask],distance[mask],borders) for k,mask in masks.items()}
            if name.startswith("vision"):
                jsonl(out/f"{label}-{name}.jsonl",(export_record(r,origins[r["run"]],p) for r,p in zip(rows,pred)))
                jsonl(out/f"{label}-{name}-next-corner.jsonl",(
                    dict(episode_id=r["run"], capture_id=r["index"],observation_time_s=r["t"]-origins[r["run"]],
                         direction=NAMES[np.argmax(sc)],distance_m=float(ds) if np.argmax(sc) else None,
                         direction_scores=dict(zip(NAMES,sc.tolist())))
                    for r,sc,ds in zip(rows,score,distance)))
    write(out/"result.json",dict(status="completed_offline_teacher_line_probe", source_revision=args.source_revision,
        script_sha256=digest(Path(__file__)), seed=None, randomness="none", termination="all evaluation groups exported",
        argv=sys.argv[1:], develop=dp, counts=dict(Counter(r["split"] for r in all_rows)), origins_host_qpc_s=origins,
        horizon_m=H.tolist(), next_horizon_m=HOLD, curvature_threshold_inv_m=CORNER, next_corner_minimum_run_m=8,
        validation_map_arc_ranges_m=VALIDATION, model_selection={k:{"alpha":p[3],"validation_lateral_mae_m":p[4]} for k,p in params.items()},
        cross_track_executed=bool(args.external), results=results))
    print(out/"result.json")


if __name__=="__main__":
    main()
