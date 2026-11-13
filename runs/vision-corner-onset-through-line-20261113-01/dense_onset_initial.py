"""Replay a dense, bilateral onset detector on stored AC camera-derived signals."""
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0, str(RUNS/"vision-corner-corrected-geometry-gate-20261112-01"))
from train_gate import (ARCHIVE, DEVELOP, SILVER, VAL, MONZA, METHODS, old, load_observer,
                        read_development, reference, frames)
from gate_inputs import SensorArrays
from replay_corner_boundary_states import observe, sensor_rows, frame_features
from corner_onset_proposals import DISTANCES

EVAL = ((SILVER,"silverstone"), (VAL,"vallelunga"), (MONZA,"monza"))
CLI = RUNS/"vision-corner-local-recovery-command-20261112-01"
SEED = 20261113
TURNS = {"left":1,"right":2}
RISK = ("outside","inside","wrong_direction","distance_error")


def profile(stations, edges, distance):
    x = (stations-distance)/8
    parts = []
    for edge in edges.T:
        use = np.isfinite(edge) & (abs(x)<=1.25)
        u, y = x[use], edge[use]
        left = int((u<0).sum())
        right = int((u>0).sum())
        if left<2 or right<2 or len(u)<5:
            parts.extend([left,right,*([np.nan]*5)])
            continue
        linear = np.column_stack((np.ones(len(u)),u))
        split = np.column_stack((linear,np.minimum(u,0)**2,np.maximum(u,0)**2))
        a = np.linalg.lstsq(split,y,rcond=None)[0]
        b = np.linalg.lstsq(linear,y,rcond=None)[0]
        c = np.linalg.lstsq(np.column_stack((linear,u*u)),y,rcond=None)[0]
        loss = np.mean((split@a-y)**2)
        parts.extend([left,right,a[2],a[3],loss,np.mean((linear@b-y)**2)-loss,
                      np.mean((np.column_stack((linear,u*u))@c-y)**2)-loss])
    return parts


def describe(data, settings):
    meta = sensor_rows(data,settings)
    edges, body, stations, retina = (data[k] for k in ("sections","motion","stations","t4t5"))
    phase = frame_features(meta,edges,body)
    valid = np.isfinite(phase).all(1) & (np.isfinite(edges).sum((1,2))>=6)
    ids = np.flatnonzero(valid)
    geometry, motion = [], []
    for i in ids:
        for direction in (1,2):
            local = edges[i] if direction==1 else -edges[i,:,::-1]
            view = retina[i,direction-1].reshape(3,4,8,8)
            # Preserve motion directions and the front/side/rear view ordering.
            motion.append(np.r_[np.sqrt(np.mean(view**2,axis=(0,3))).ravel(),
                                np.sqrt(np.mean((view[-1]-view[0])**2,axis=-1)).ravel()])
            for distance in DISTANCES:
                geometry.append([distance,body[i,2]*(1 if direction==1 else -1),*phase[i],
                                 *profile(stations,local,distance)])
    g = np.asarray(geometry).reshape(len(ids),2,len(DISTANCES),48)
    v = np.asarray(motion).reshape(len(ids),2,64)
    combined = np.concatenate((g,np.broadcast_to(v[:,:,None,:],(*g.shape[:3],64))),axis=-1)
    return ids, (g,combined)


def targets(rows, ids):
    y = np.zeros((len(ids),2,len(DISTANCES)),bool)
    for i, frame in enumerate(ids):
        row = rows[frame]
        if row["truth"]=="approach":
            distance = np.asarray(row["truth_endpoints_m"])
            y[i,int(row["label"])-1] = np.max(abs(DISTANCES[:,None]-distance),axis=1)<=5.
    return y


def fit(groups, method):
    x,y,w = [],[],[]
    for _,values,labels in groups:
        features = values[method]
        use = np.ones(len(labels),bool) if method==0 else np.isfinite(features[:,0,0,-64:]).all(1)
        xx, yy = features[use],labels[use]
        x.append(xx.reshape(-1,xx.shape[-1]))
        y.append(yy.ravel())
        w.append(np.where(yy, .5/max(1,yy.sum()), .5/max(1,(~yy).sum())).ravel())
    x,y,w = map(np.concatenate,(x,y,w))
    clf = HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,min_samples_leaf=35,
              l2_regularization=2.,learning_rate=.075,early_stopping=False,random_state=SEED)
    return clf.fit(x,y,sample_weight=w/w.mean())


def best_cells(group, model, method):
    ids, features = group[:2]
    x = features[method]
    score = model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(len(ids),-1)
    order = score.argmax(1)
    side, distance = order//len(DISTANCES)+1,DISTANCES[order%len(DISTANCES)]
    confidence = score[np.arange(len(ids)),order]
    if method:
        confidence[~np.isfinite(x[:,0,0,-64:]).all(1)] = -1.
    return ids,side,distance,confidence


def label(row, direction, distance):
    category = old.category(row,(int(direction),float(distance)))
    if category in ("out_of_range_alert","straight_alert"):
        return "outside"
    if category=="in_corner_alert":
        return "inside"
    if category=="joint_hit":
        return "hit_"+row["label"]
    return category


def choose_cut(calibration, cap):
    score = np.concatenate([v[0] for v in calibration])
    categories = np.concatenate([v[1] for v in calibration])
    best = None
    for cut in np.r_[np.unique(score[score>=0]),1.01]:
        count = Counter(categories[score>=cut])
        if any(count[k]>cap[k] for k in RISK):
            continue
        left,right = count["hit_1"],count["hit_2"]
        key = (min(left,right),left+right,-sum(count[k] for k in RISK),float(cut))
        if best is None or key>best[0]:
            best = key,dict(count)
    return best[0][-1],best[1]


def turning(row):
    return any(f["state"] in TURNS for f in row["forecasts"].values())


def score_archive(labels, outputs, method):
    counts = Counter()
    for row in labels:
        f = outputs[row["run"],row["frame"]]["forecasts"][method]
        side = TURNS.get(f["state"])
        key = label(row,side,f["distance_m"]) if side else ("miss_"+row["label"] if row["truth"]=="approach" else "quiet")
        counts[key]+=1
        if side and row["truth"]=="approach" and int(row["label"])==side:
            counts["direction_"+row["label"]]+=1
    return counts


def run():
    models, settings, paths = load_observer()
    paths.update((Path(__file__),Path(read_development.__code__.co_filename)))
    dev, references, control = {},{},{}
    for track in DEVELOP:
        data, rows, _ = read_development(track,paths,settings)
        data = SensorArrays(data)
        ids, values = describe(data,settings)
        dev[track] = ids,values,targets(rows,ids)
        references[track] = rows
        control[track] = observe(data,models,settings)[0]
    fitted = [fit(list(dev.values()),m) for m in (0,1)]
    folds = [[],[]]
    for track,g in dev.items():
        train = [v for t,v in dev.items() if t!=track]
        for m in (0,1):
            ids, side, dist, confidence = best_cells(g,fit(train,m),m)
            eligible = np.array([not turning(control[track][i]) for i in ids])
            cats = np.array([label(references[track][i],s,d) for i,s,d in zip(ids,side,dist,strict=True)])
            folds[m].append((confidence[eligible],cats[eligible]))
        print("Development fold",track,flush=True)
    # Same bounded allowance per readout, chosen before opening evaluation labels.
    cap = dict(outside=8,inside=5,wrong_direction=2,distance_error=2)
    cuts = [choose_cut(fold,cap) for fold in folds]
    checkpoint = OUT/"dense-onset.joblib"
    joblib.dump(fitted,checkpoint)
    frozen = OUT/"onset-training.json"
    old.write(frozen,dict(development=list(dev),excluded=[t for t,_ in EVAL],cuts=cuts,additional_risk_cap=cap,
              seed=SEED,checkpoint_sha256=old.digest(checkpoint),source_revision=old.revision(),
              source_sha256={str(p):old.digest(p) for p in sorted(paths)},termination="frozen_before_replay"))
    print("Frozen",cuts,flush=True)
    output={}
    for track,folder in EVAL:
        root = CLI/("vallelunga-retry" if track==VAL else folder)
        source = root/"recovered-sensors.npz"
        receipt = old.read(root/"result.json")
        assert old.digest(source)==receipt["recovered_input_sha256"]
        with np.load(source,allow_pickle=False) as f:
            sensor = SensorArrays({k:f[k] for k in f.files})
        baseline_path = root/"predictions/forecasts.jsonl"
        base = frames(baseline_path,old.read(baseline_path.with_name("result.json"))["forecasts_sha256"])
        g = describe(sensor,settings)
        a,b = [best_cells(g,fitted[m],m) for m in (0,1)]
        np.testing.assert_array_equal(a[0],b[0])
        final = [dict(row,forecasts=dict(row["forecasts"])) for row in base]
        for j,i in enumerate(a[0]):
            if turning(base[i]):
                continue
            # A common geometry offer isolates T4/T5 confirmation in the paired run.
            # T4/T5 can confirm only if its strongest cell agrees with that offer.
            side,distance = int(a[1][j]),float(a[2][j])
            accepted = a[3][j]>=cuts[0][0], (a[1][j]==b[1][j] and a[2][j]==b[2][j] and b[3][j]>=cuts[1][0])
            if any(accepted):
                final[i]["candidate"] = [side,distance]
                for method,on in zip(METHODS,accepted,strict=True):
                    if on:
                        final[i]["forecasts"][method] = dict(state=("left","right")[side-1],distance_m=distance)
        labels, truth_path = reference(track)
        trace = OUT/f"{track}-onset.jsonl"
        old.jsonl(trace,final)
        index = lambda rows:{(r["run"],r["frame"]):r for r in rows}
        output[track] = dict(baseline={m:score_archive(labels,index(base),m) for m in METHODS},
                             revised={m:score_archive(labels,index(final),m) for m in METHODS},
                             source_sha256={str(p):old.digest(p) for p in (source,baseline_path,truth_path)},
                             forecasts_sha256=old.digest(trace),frames=len(final))
        print(track,output[track]["baseline"],output[track]["revised"],flush=True)
    old.write(OUT/"onset-result.json",dict(tracks=output,training_sha256=old.digest(frozen),termination="archive_end"))


if __name__=="__main__":
    run()
