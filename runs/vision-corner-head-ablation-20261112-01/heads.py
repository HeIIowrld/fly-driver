"""Compare direction and presence replacement on held-out development circuits."""
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import anchor_distance as anchor
from evaluate_fused_corner import frame_key, load_score, read_rows
from infer_frozen_next_corner import load_features, load_readout
from pair_audit import CALIBRATION_TRACKS, TRACKS, load, motion_features
from score_corner_captures import forecast_state, summarize
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import score

OUT = Path(__file__).parent
SILVER = "ks_silverstone1967"
SILVER_RESULT = Path("runs/vision-silverstone1967-independent-evaluation-20261112-01")
PROTOCOL = Path("runs/vision-silverstone1967-independent-protocol-20261111-01/base.json")
CATEGORIES = ("approach", "straight", "out_of_range", "in_corner")
ARMS = ("frozen_heads", "presence_only", "direction_only", "both")
CUTS = (.35, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95)
SEED = 31
REFLECT = np.array([0, 2, 1])


def load_silver(hashes):
    base = SILVER_RESULT/"visual/base"
    root = SILVER_RESULT/"visual"
    base_receipt, proof = read(base/"result.json"), read(root/"result.json")
    original = load_score(base/"parent/score", base_receipt["score_receipt_sha256"]["parent"])
    rows = load_score(root/"score", proof["score_receipt_sha256"])
    rules = read(PROTOCOL)["comparison"]
    pins = Path(rules["candidate_pins"])
    index = base/"features/index.jsonl"
    direct = base/"features/original/features.npz"
    reflected = base/"features/reflected/features.npz"
    assert [frame_key(r) for r in rows] == [frame_key(r) for r in read_rows(index)]
    a, av, _ = load_features(direct, digest(index), len(rows))
    b, bv, _ = load_features(reflected, digest(index), len(rows))
    valid = av & bv
    infer, threshold = load_readout(read(pins))
    logits, distance, presence = infer(a[valid], b[valid], return_scores=True)
    a, b = motion_features(a[valid]), motion_features(b[valid])
    shared, odd = .5*(a+b), .5*(a-b)
    x = np.stack((np.c_[shared,odd], np.c_[shared,-odd]), axis=1)
    hashes.update((PROTOCOL, pins, index, direct, reflected, base/"result.json", root/"result.json"))
    return dict(rows=rows, original=original, rules=rules, valid=valid,
                selected=[r for r,ok in zip(rows,valid,strict=True) if ok], x=x,
                scores=logits, distance=distance, presence=presence, presence_threshold=threshold)


def examples(group):
    records = group["selected"]
    selected = np.array([row["truth"] in CATEGORIES for row in records])
    selected_rows = [r for r,ok in zip(records,selected,strict=True) if ok]
    labels = np.array([int(r["label"]) if r["truth"] == "approach" else 0 for r in selected_rows])
    keys = [(r["label"], r["physical_corner"]) for r in selected_rows]
    corners = Counter(key for key, y in zip(keys,labels,strict=True) if y)
    sides = Counter(side for side, _ in corners)
    negatives = Counter(r["truth"] for r,y in zip(selected_rows,labels,strict=True) if not y)
    weight = np.array([
        .5 / len(sides) / sides[key[0]] / corners[key] if y
        else .5 / len(negatives) / negatives[row["truth"]]
        for row,key,y in zip(selected_rows,keys,labels,strict=True)])
    x = group["x"][selected]
    return np.concatenate((x[:,0],x[:,1])), np.r_[labels, REFLECT[labels]], np.tile(weight*.5,2)


def render(group, directions, presence, cutoff):
    known = iter(zip(directions, presence, group["distance"], strict=True))
    rows = []
    for row, usable in zip(group["rows"],group["valid"],strict=True):
        if not usable:
            rows.append(row)
            continue
        direction, p, distance = next(known)
        direction, distance = int(direction), float(distance)
        prediction = dict(available=True, direction=direction, distance_m=distance if direction else None,
                          alert=bool(direction and 0. <= distance <= 20. and p>=cutoff))
        state = forecast_state(prediction)
        if prediction["alert"]:
            rows.append(dict(score(row,(direction,distance)),abstain_reason=None,presence_score=float(p)))
        else:
            reason = ("distance_out_of_range" if distance > 20. else "presence_rejected") if state=="abstain" else None
            rows.append(dict(row,state=state,predicted_direction=direction,
                             predicted_distance_m=prediction["distance_m"],presence_score=float(p),
                             abstain_reason=reason,direction_hit=False,joint_hit=False,
                             signed_distance_error_m=None,absolute_distance_error_m=None))
    assert next(known,None) is None
    return rows


def review(group, rows):
    summary = anchor.evaluate(group,rows)
    runs = [summarize([r for r in rows if r["run"]==run]) for run in dict.fromkeys(r["run"] for r in rows)]
    summary["false"] = {key:sum(s["false_alerts"][key]["frames"] for s in runs)
                         for key in ("straight","out_of_range")}
    summary["outputs"] = {key:sum(s["outputs"][key] for s in runs) for key in runs[0]["outputs"]}
    summary["stable_physical"] = len({event["physical_corner"] for event in summary["stable_events"]})
    return summary


def old_directions(scores):
    winners = np.argmax(scores,axis=1)
    winners[np.sum(scores == scores.max(axis=1,keepdims=True),axis=1)>1] = 0
    return winners


def feasible(old, new):
    a,b = old["metrics"]["candidate"],new["metrics"]["candidate"]
    return (all(new["false"][key] <= n for key,n in old["false"].items())
            and b["missing"] <= a["missing"] and b["left"] >= a["left"] and b["right"] >= a["right"]
            and b["joint"] >= a["joint"] and new["stable_physical"] >= old["stable_physical"])


def main():
    hashes = {Path(__file__),Path(anchor.__file__),Path("scripts/score_corner_captures.py"),
              Path("scripts/infer_frozen_next_corner.py")}
    tracks = (*TRACKS,SILVER)
    groups = {t:load(t,hashes) for t in TRACKS}
    groups[SILVER] = load_silver(hashes)
    samples = {t:examples(groups[t]) for t in CALIBRATION_TRACKS}
    baseline = {t:review(g,render(g,old_directions(g["scores"]),g["presence"],g["presence_threshold"]))
                for t,g in groups.items()}

    @lru_cache
    def fit(names):
        x,y,w = (np.concatenate(parts) for parts in zip(*(samples[t] for t in names),strict=True))
        w *= len(w)/w.sum()
        scale = StandardScaler().fit(x,sample_weight=w)
        x = scale.transform(x).clip(-8.,8.)
        presence = LogisticRegression(C=.08,random_state=SEED,max_iter=600).fit(x,y!=0,sample_weight=w)
        direction = LogisticRegression(C=.08,random_state=SEED,max_iter=600).fit(x,y,sample_weight=w)
        assert np.array_equal(direction.classes_,[0,1,2])
        return scale,presence,direction

    @lru_cache
    def predictions(track,names):
        assert track not in names and all(t in CALIBRATION_TRACKS for t in names)
        scale,presence,direction = fit(names)
        x = groups[track]["x"]
        z = [scale.transform(x[:,i]).clip(-8.,8.) for i in range(2)]
        p = .5*(presence.predict_proba(z[0])[:,1]+presence.predict_proba(z[1])[:,1])
        logits = .5*(direction.predict_proba(z[0])+direction.predict_proba(z[1])[:,REFLECT])
        return old_directions(logits),p

    def apply(track, names, arm, cutoff):
        g = groups[track]
        d,p = predictions(track,names)
        if arm=="presence_only":
            d = old_directions(g["scores"])
        if arm=="direction_only":
            p,cutoff = g["presence"],g["presence_threshold"]
        return render(g,d,p,cutoff)

    records,details = {},[]
    for track in tracks:
        names = tuple(t for t in CALIBRATION_TRACKS if t!=track)
        nested = {t:tuple(k for k in names if k!=t) for t in names}
        result = {"frozen_heads":dict(selected=baseline[track])}
        for arm in ARMS[1:]:
            if arm=="direction_only":
                rows = apply(track,names,arm,None)
                result[arm] = dict(selected=review(groups[track],rows),cutoff="frozen_presence")
            else:
                choices = []
                for cut in CUTS:
                    r = {t:review(groups[t],apply(t, peers,arm,cut)) for t,peers in nested.items()}
                    if all(feasible(baseline[t],v) for t,v in r.items()):
                        gains = tuple(sum(r[t]["metrics"]["candidate"][k]-baseline[t]["metrics"]["candidate"][k]
                                          for t in names) for k in ("left","right","joint"))
                        stable = sum(r[t]["stable_physical"]-baseline[t]["stable_physical"] for t in names)
                        if any(gains) or stable:
                            choices.append(((stable,gains[0],gains[2],gains[1],cut),cut))
                picked = max(choices)[1] if choices else None
                diagnostic_cut = picked if picked is not None else .50
                rows = apply(track,names,arm,diagnostic_cut)
                result[arm] = dict(cutoff=picked,diagnostic_cutoff=diagnostic_cut,
                    selected=review(groups[track],rows) if picked is not None else None,
                    sweep={str(c):review(groups[track],apply(track,names,arm,c)) for c in CUTS})
            details.extend(dict(track=track,arm=arm,run=r["run"],frame=r["frame"],state=r["state"],
                label=r["label"],direction_hit=r["direction_hit"],joint_hit=r["joint_hit"],
                distance_m=r["predicted_distance_m"]) for r in rows if r["truth"]=="approach")
        records[track] = dict(frames=len(groups[track]["rows"]),training=names,calibration=nested,arms=result)
        print(track,{arm:r.get("cutoff") for arm,r in result.items()},flush=True)
    path=OUT/"approach-rows.jsonl"
    jsonl(path,details)
    write(OUT/"result.json",dict(status="evaluated",seed=SEED,tracks=records,rows_sha256=digest(path),
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
