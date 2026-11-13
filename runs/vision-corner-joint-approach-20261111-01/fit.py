"""Evaluate joint boundary and T4/T5 approach readout with nested circuit splits."""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from audit import INPUTS, MUGELLO, RECOVERY, RUNS
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows
from infer_frozen_next_corner import load_features
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import BASES, compare_tracks, score

OUT = Path(__file__).parent
BOUNDARY = RUNS/"vision-corner-onset-support-20261111-01"
ASSOCIATION = RUNS/"vision-corner-motion-association-20261111-01"
MODES = ("geometry","motion","joint")
THRESHOLDS = (.50,.60,.70,.80,.85,.90,.925,.95,.975,.99,1.)

def motion_features(values):
    """Retain quadrant motion at the latest time and its two causal differences."""
    grid = values.astype(float).reshape(-1,3,4,8,2,2,2,2)
    grid = grid.mean(axis=(5,7))
    scale = np.sqrt(np.mean(grid**2,axis=(1,3,4,5),keepdims=True)).clip(.025)
    grid /= scale
    latest, prior, oldest = grid[:,2],grid[:,1],grid[:,0]
    return np.stack((latest,latest-prior,prior-oldest),axis=1).reshape(len(grid),-1)

def geometry_features(row, proposals, side):
    sign = 1 if side==1 else -1
    raw = dict(row.get("curvature_samples",()))
    observed = set(row.get("available_sections_m",()))
    profile = np.array([sign*raw.get(float(x),0.) for x in range(4,22)])
    support = [float(x in raw) for x in range(4,22)]
    stations = [float(x in observed) for x in range(2,25,2)]
    pool = [d for direction,d in proposals.get("pooled_all") or () if direction==side]
    option = proposals.get("associated")
    return np.r_[profile, support, stations,
                 [float(x in pool) for x in range(4,21,2)],
                 float(bool(option and option[0]==side)),
                 option[1] if option and option[0]==side else 0.]

def load_track(track, paths):
    config_path = RECOVERY/INPUTS[track][0]
    config = load_frozen(config_path)
    rules_path = Path(config["base_protocol"])
    rules = read(rules_path)["comparison"]
    evaluation = MUGELLO if track=="mugello" else RECOVERY/track
    proof_path = evaluation/"result.json"
    proof = read(proof_path)
    assert proof["protocol_sha256"]==digest(config_path)
    current = load_score(evaluation/"score",proof["score_receipt_sha256"])
    base = BASES[track]
    base_path = base/"result.json"
    assert proof["base_receipt_sha256"]==digest(base_path)
    parent = load_score(base/"parent/score",read(base_path)["score_receipt_sha256"]["parent"])
    geo = BOUNDARY/track/"rows.jsonl"
    association = ASSOCIATION/track/"rows.jsonl"
    for p in (geo,association):
        assert digest(p)==read(p.with_name("result.json"))["rows_sha256"]
    index = base/"features/index.jsonl"
    direct = base/"features/original/features.npz"
    reflected = base/"features/reflected/features.npz"
    index_rows = read_rows(index)
    x,a,_ = load_features(direct,digest(index),len(index_rows))
    y,b,_ = load_features(reflected,digest(index),len(index_rows))
    paths.update((config_path,rules_path,proof_path,base_path,geo,association,index,direct,reflected))
    masks = a & b
    geom = read_rows(geo)
    assoc = read_rows(association)
    for rows in zip(current,index_rows,geom,assoc,strict=True):
        assert len({frame_key(r) for r in rows})==1
    m = np.stack((motion_features(x[masks]),motion_features(y[masks])),axis=1)
    g = np.asarray([[geometry_features(row,p,side) for side in (1,2)]
                    for row,p,ok in zip(geom,assoc,masks,strict=True) if ok])
    indices = np.flatnonzero(masks)
    selected = [current[i] for i in indices]
    labels = np.array([[r["truth"]=="approach" and r["label"]==str(side) for side in (1,2)] for r in selected])
    train = np.array([r["truth"] in ("approach","in_corner","straight","out_of_range") for r in selected])
    distances = np.array([r["truth_distance_m"] if r["truth"]=="approach" else 0. for r in selected])
    return dict(current=current,parent=parent,rules=rules,indices=indices,labels=labels,train=train,
                distances=distances,revision=proof["revision"], features={"geometry":g,"motion":m,"joint":np.concatenate((g,m),axis=2)})

def fit_model(groups, mode):
    x,y,d,weights = [],[],[],[]
    for group in groups:
        use = group["train"]
        targets = group["labels"][use].reshape(-1)
        data = group["features"][mode][use]
        x.append(data.reshape(-1,data.shape[-1]))
        y.append(targets)
        d.append(np.repeat(group["distances"][use],2))
        weights.append(np.where(targets, .5/targets.sum(), .5/(~targets).sum()))
    x,y,d,w = np.concatenate(x),np.concatenate(y),np.concatenate(d),np.concatenate(weights)
    w *= len(w)/sum(w)
    scaler = StandardScaler().fit(x,sample_weight=w)
    z = np.clip(scaler.transform(x),-6.,6.)
    presence = LogisticRegression(C=.05,max_iter=600,tol=1e-5).fit(z,y,sample_weight=w)
    distance = Ridge(alpha=100.).fit(z[y],d[y],sample_weight=w[y])
    return scaler,presence,distance

def predictions(model, group, mode):
    scaler,presence,distance = model
    features = group["features"][mode]
    flat = np.clip(scaler.transform(features.reshape(-1,features.shape[-1])),-6.,6.)
    probs = presence.predict_proba(flat)[:,1].reshape(-1,2)
    dist = distance.predict(flat).reshape(-1,2).clip(4.,20.)
    return {int(index):(tuple(map(float,p)),tuple(map(float,d))) for index,p,d in zip(group["indices"],probs,dist,strict=True)}

def apply(group, forecasts, threshold, repeated):
    result, changes = [],[]
    previous = None
    previous_side = None
    for index,row in enumerate(group["current"]):
        pair = forecasts.get(index)
        side = None
        if pair:
            p,d = pair
            if max(p)>=threshold and abs(p[0]-p[1])>1e-10:
                side = 1 if p[0]>p[1] else 2
        qualified = side is not None and (not repeated or (previous is not None and adjacent(previous,row)
                                                          and previous_side==side))
        if row["state"]=="straight" and qualified:
            row = score(row,(side,d[side-1]))
            changes.append(row)
        result.append(row)
        previous,previous_side = row,side
    return result,changes

def score_group(group, forecasts, threshold, repeated):
    result,changes = apply(group,forecasts,threshold,repeated)
    return compare_tracks(group["parent"],result,group["rules"]),changes

def threshold_search(groups, calibration, repeated):
    for threshold in THRESHOLDS:
        reviews = [score_group(groups[name],scores,threshold,repeated)[0] for name,scores in calibration.items()]
        yield threshold,reviews

def select_threshold(groups, calibration, repeated):
    references = [compare_tracks(groups[name]["parent"],groups[name]["current"],groups[name]["rules"])
                  for name in calibration]
    best = None
    for threshold,reviews in threshold_search(groups,calibration,repeated):
        protected = True
        for r, ref in zip(reviews,references,strict=True):
            current = r["metrics"]["candidate"]
            base = ref["metrics"]["candidate"]
            original = ref["metrics"]["parent"]
            protected &= (current["outside"]<=max(base["outside"],original["outside"])
                          and current["missing"]<=base["missing"]
                          and current["right"]>=base["right"] and current["joint"]>=base["joint"])
        if not protected:
            continue
        gained = sum(r["metrics"]["candidate"]["left"]-r["metrics"]["parent"]["left"] for r in reviews)
        joints = sum(r["metrics"]["candidate"]["joint"] for r in reviews)
        false = sum(r["metrics"]["candidate"]["outside"] for r in reviews)
        rank = (gained,joints,-false,threshold)
        if best is None or rank>best[0]:
            best = rank,threshold
    assert best is not None
    return best[1]


def run():
    paths = {Path(__file__)}
    groups = {name:load_track(name,paths) for name in INPUTS}
    details, change_rows = {},[]
    for name,group in groups.items():
        train = {key:g for key,g in groups.items() if key!=name}
        result = dict(revision=group["revision"], before=compare_tracks(group["parent"],group["current"],group["rules"]),
                      methods={})
        for mode in MODES:
            calibration = {key:predictions(fit_model([g for n,g in train.items() if n!=key],mode),held,mode)
                           for key,held in train.items()}
            fitted = fit_model(list(train.values()),mode)
            forecasts = predictions(fitted,group,mode)
            methods = {}
            for repeated in (False,True):
                method = "two_frames" if repeated else "direct"
                threshold = select_threshold(train,calibration,repeated)
                report,changes = score_group(group,forecasts,threshold,repeated)
                methods[method] = dict(threshold=threshold,comparison=report,
                                       added=len(changes))
                change_rows.extend(dict(row,held_circuit=name,feature_mode=mode,method=method) for row in changes)
            methods["diagnostic_thresholds"] = {
                str(threshold): reviews[0]["metrics"]["candidate"]
                for threshold,reviews in threshold_search({name:group},{name:forecasts},False)
            }
            result["methods"][mode] = methods
        details[name] = result
        print(name, {mode:{k:(v["threshold"],v["added"],v["comparison"]["metrics"]["candidate"])
                           for k,v in a.items() if k!="diagnostic_thresholds"} for mode,a in result["methods"].items()},flush=True)
    changes = OUT/"changes.jsonl"
    jsonl(changes,change_rows)
    write(OUT/"result.json",dict(tracks=details, seed=None, changes_sha256=digest(changes),
          source_sha256={str(p):digest(p) for p in sorted(paths)},termination="nested_circuit_evaluation_completed"))

if __name__=="__main__":
    run()
