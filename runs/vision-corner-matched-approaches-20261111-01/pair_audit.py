"""Audit Spa direction gates and compare same-corner distance-pair training."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from audit import INPUTS, MUGELLO, RECOVERY, RUNS
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows
from fit import motion_features
from infer_frozen_next_corner import load_features, load_readout
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import BASES, compare_tracks, score

OUT = Path(__file__).parent
TRACKS = (*INPUTS, "imola", "spa")
CALIBRATION_TRACKS = tuple(track for track in TRACKS if track != "spa")
LIMITS = (10.,12.,14.,16.,18.,20.,None)


def sources(track):
    if track == "spa":
        root = RUNS/"vision-spa-independent-evaluation-20261111-01"
        final = root/"comparison"
        config = RUNS/"vision-spa-independent-protocol-20261111-01/candidate.json"
        return final,final/"base",config
    if track == "imola":
        final = RUNS/"vision-imola-approach-diagnostic-20261111-01"
        return final,final/"base",RUNS/"vision-imola-approach-capture-protocol-20261111-01/diagnostic-recovery.json"
    return MUGELLO if track == "mugello" else RECOVERY/track,BASES[track],RECOVERY/INPUTS[track][0]


def load(track, hashes):
    final, base, protocol_path = sources(track)
    protocol = load_frozen(protocol_path)
    rules_path = Path(protocol["base_protocol"])
    rules = read(rules_path)["comparison"]
    final_path, base_path = final/"result.json",base/"result.json"
    proof, base_proof = read(final_path),read(base_path)
    assert proof["protocol_sha256"] == digest(protocol_path)
    assert proof["base_receipt_sha256"] == digest(base_path)
    original = load_score(base/"parent/score",base_proof["score_receipt_sha256"]["parent"])
    current = load_score(final/"score",proof["score_receipt_sha256"])
    index = base/"features/index.jsonl"
    direct = base/"features/original/features.npz"
    mirror = base/"features/reflected/features.npz"
    rows = read_rows(index)
    x,available_x,_ = load_features(direct,digest(index),len(rows))
    y,available_y,_ = load_features(mirror,digest(index),len(rows))
    assert all(frame_key(a)==frame_key(b) for a,b in zip(current,rows,strict=True))
    valid = available_x & available_y
    hashes.update((protocol_path,rules_path,final_path,base_path,index,direct,mirror))
    pins = Path(rules["candidate_pins"])
    hashes.add(pins)
    infer,threshold = load_readout(read(pins))
    scores, distance, presence = infer(x[valid],y[valid],return_scores=True)
    a,b = motion_features(x[valid]),motion_features(y[valid])
    shared,odd = .5*(a+b),.5*(a-b)
    features = np.stack((np.c_[shared,odd],np.c_[shared,-odd]),axis=1)
    selected = [r for r,ok in zip(current,valid,strict=True) if ok]
    before = compare_tracks(original,current,rules)
    assert before["metrics"] == proof["comparison"]["metrics"]
    return dict(rows=current,original=original,rules=rules,valid=valid,selected=selected,x=features,
                scores=scores,distance=distance,presence=presence,presence_threshold=threshold,
                before=before,revision=proof["revision"])


def audit_spa(group):
    details=[]
    stats={}
    for label in ("1","2"):
        missed=Counter()
        for row,scores,distance,presence in zip(group["selected"],group["scores"],group["distance"],group["presence"],strict=True):
            if row["truth"]!="approach" or row["label"]!=label or row["direction_hit"]:
                continue
            side=int(np.argmax(scores[1:]))+1
            flags=dict(background_won=bool(scores[0]>=max(scores[1:])),correct_among_sides=side==int(label),
                       presence_pass=bool(presence>=group["presence_threshold"]),range_pass=bool(distance<=20.))
            missed["available"]+=1
            for key,value in flags.items():
                missed[key]+=value
            missed["side_with_both_gates"]+=flags["correct_among_sides"] and flags["presence_pass"] and flags["range_pass"]
            details.append(dict(run=row["run"],frame=row["frame"],corner=row["physical_corner"],label=label,
                                scores=scores.tolist(),distance_m=float(distance),presence=float(presence),**flags))
        stats[label]=missed
    path=OUT/"spa-missed-frames.jsonl"
    jsonl(path,details)
    return dict(summary=stats,rows_sha256=digest(path))


def make_pairs(group):
    """Pair each near view with at most one far view per target range from the same approach."""
    previous=defaultdict(list)
    near_rows,far_rows,side_rows,deltas=[],[],[],[]
    for index,row in enumerate(group["selected"]):
        if row["physical_corner"] is None or row["label"] not in ("1","2"):
            continue
        k=(row["run"],row["generation"],row["physical_corner"],row["label"])
        d=row["truth_distance_m"]
        if row["truth"]=="out_of_range" and d is not None and 20<d<=52.:
            previous[k].append((index,row["received_s"],d))
        if row["truth"]!="approach":
            continue
        earlier=[(i,t,v) for i,t,v in previous[k] if 0<row["received_s"]-t<=12.]
        chosen={}
        for target in (24.,36.,48.):
            if earlier:
                entry=min(earlier,key=lambda p:abs(p[2]-target))
                if abs(entry[2]-target)<=7.:
                    chosen[entry[0]]=entry[2]
        for other,far_distance in chosen.items():
            near_rows.append(index)
            far_rows.append(other)
            side_rows.append(int(row["label"])-1)
            deltas.append((d,far_distance))
    return (np.array(near_rows,dtype=int),np.array(far_rows,dtype=int),np.array(side_rows,dtype=int),
            np.array(deltas,dtype=float).reshape(-1,2))


def fit_model(groups, pairwise):
    a,b,d,weights=[],[],[],[]
    for group in groups:
        near,far,sides,targets=group["pairs"]
        if len(targets):
            a.append(group["x"][near,sides])
            b.append(group["x"][far,sides])
            d.append(targets)
            weights.append(np.full(len(targets),1./len(targets)))
    a,b,d,w=np.concatenate(a),np.concatenate(b),np.concatenate(d),np.concatenate(weights)
    w *= len(w)/w.sum()
    scaler=StandardScaler().fit(np.r_[a,b],sample_weight=np.r_[w,w])
    a=np.clip(scaler.transform(a),-6,6)
    b=np.clip(scaler.transform(b),-6,6)
    if pairwise:
        model=Ridge(alpha=300.,fit_intercept=False).fit(b-a,d[:,1]-d[:,0],sample_weight=w)
        offset=float(np.average(d.ravel(order="F")-model.predict(np.r_[a,b]),weights=np.r_[w,w]))
    else:
        model=Ridge(alpha=300.).fit(np.r_[a,b],d.ravel(order="F"),sample_weight=np.r_[w,w])
        offset=0.
    return scaler,model,offset


def predict(group, model):
    scaler,regressor,offset=model
    data=group["x"]
    x=np.clip(scaler.transform(data.reshape(-1,data.shape[-1])),-6,6)
    return (regressor.predict(x)+offset).reshape(-1,2)


def pair_metrics(group, predicted):
    near,far,sides,distances=group["pairs"]
    a,b=predicted[near,sides],predicted[far,sides]
    errors=np.abs((b-a)-(distances[:,1]-distances[:,0]))
    return dict(pairs=len(errors),correct_order=int(np.sum(b>a)),mae_delta_m=float(errors.mean()) if len(errors) else None,
                near_mae_m=float(np.abs(a-distances[:,0]).mean()) if len(errors) else None)


def apply(group, predicted, limit):
    valid=iter(zip(group["scores"],group["presence"],predicted,strict=True))
    out=[]
    for row,ok in zip(group["rows"],group["valid"],strict=True):
        if ok:
            scores,presence,distances=next(valid)
            side=int(np.argmax(scores[1:]))+1
            if (limit is not None and row["state"] in ("straight","abstain")
                    and (row["state"]=="straight" or row["abstain_reason"]=="presence_rejected")
                    and abs(scores[1]-scores[2])>1e-10
                    and presence>=group["presence_threshold"] and distances[side-1]<=limit):
                row=score(row,(side,float(np.clip(distances[side-1],4.,20.))))
        out.append(row)
    return compare_tracks(group["original"],out,group["rules"])


def choose(calibration, groups):
    ranked=[]
    for limit in LIMITS:
        reports=[apply(groups[k],p,limit) for k,p in calibration.items()]
        if any(r["metrics"]["candidate"]["outside"]>max(
            groups[k]["before"]["metrics"][name]["outside"] for name in ("parent","candidate"))
            for k,r in zip(calibration,reports,strict=True)):
            continue
        n=lambda metric:sum(r["metrics"]["candidate"][metric] for r in reports)
        ranked.append(((n("left"),n("joint"),-n("outside"),-float(limit or 0.)),limit))
    return max(ranked)[1]


def run():
    hashes={Path(__file__),Path("runs/vision-corner-joint-approach-20261111-01/fit.py")}
    groups={track:load(track,hashes) for track in TRACKS}
    diagnostic=audit_spa(groups["spa"])
    for group in groups.values():
        group["pairs"]=make_pairs(group)
    results={}
    for pairwise in (False,True):
        mode="paired" if pairwise else "absolute"
        model_cache={}
        for size in (3,4,5):
            for keys in combinations(CALIBRATION_TRACKS,size):
                model_cache[keys]=fit_model([groups[k] for k in keys],pairwise)
        results[mode]={}
        for track,group in groups.items():
            train=tuple(k for k in CALIBRATION_TRACKS if k!=track)
            calibration={k:predict(groups[k],model_cache[tuple(other for other in train if other!=k)]) for k in train}
            limit=choose(calibration,groups)
            pred=predict(group,model_cache[train])
            report=apply(group,pred,limit)
            pm=pair_metrics(group,pred)
            results[mode][track]=dict(pairs=pm,limit_m=limit,comparison=report,revision=group["revision"])
            print(mode,track,pm,limit,report["metrics"]["candidate"],flush=True)
    write(OUT/"result.json",dict(tracks=results,spa_frozen_stage_audit=diagnostic,
          scope="post-evaluation development study; Spa labels used in diagnostics only",
          source_sha256={str(p):digest(p) for p in sorted(hashes)},seed=None,termination="same_corner_pair_comparison_complete"))

if __name__=="__main__":
    run()
