"""Fit candidate likelihood and within-frame preference with circuit exclusion."""
from collections import Counter
from pathlib import Path

import numpy as np

from alternatives import OUT, DEVELOP, SILVER, VAL, data, old
from fit_gate import SEED, FAR_RATE, VALID_CATEGORIES, budget_cut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import joblib

MODELS = ("static","temporal","t4t5")
METHODS = ("static","temporal","t4t5_confirm","t4t5_rerank")
RANK_WEIGHT = .35


def geometry(group,mode):
    if mode=="static":
        return group["edge"]
    return np.c_[group["edge"],group["history"]]


def encode(group,model,mode):
    x = np.clip(model["geometry"].transform(geometry(group,mode)),-6.,6.)
    if mode=="t4t5":
        m = np.clip(model["motion"].transform(group["motion"]),-6.,6.)
        x = np.c_[x,np.clip(model["pca"].transform(m),-6.,6.)]
    return x


def fit(groups,mode):
    weights, targets, features, motion = [],[],[],[]
    pair_index = []
    offset = 0
    for g in groups:
        keep = np.flatnonzero(np.isin(g["errors"],VALID_CATEGORIES))
        frames = g["index"][keep]
        target = g["target"][keep]
        count = np.bincount(frames)
        w = 1./count[frames]
        for condition in (False,True):
            take = target==condition
            if take.any():
                w[take] *= .5/w[take].sum()
        groups_by_frame = np.split(np.arange(len(frames)),np.flatnonzero(np.diff(frames))+1)
        for local in groups_by_frame:
            positive,negative = local[target[local]],local[~target[local]]
            for i in positive:
                # Prefer alternatives whose reported distance is close to the positive.
                rank = sorted(negative,key=lambda j:abs(g["offers"][keep[i]][1]-g["offers"][keep[j]][1]))
                pair_index.extend((offset+i,offset+j) for j in rank[:2])
        offset += len(keep)
        features.append(geometry(g,mode)[keep])
        motion.append(g["motion"][keep])
        weights.append(w)
        targets.append(target)
    raw, x_motion, w, y = (np.concatenate(a) for a in (features,motion,weights,targets))
    w *= len(w)/w.sum()
    scale = StandardScaler().fit(raw,sample_weight=w)
    x = np.clip(scale.transform(raw),-6.,6.)
    if mode=="t4t5":
        scaler = StandardScaler().fit(x_motion)
        m = np.clip(scaler.transform(x_motion),-6.,6.)
        pca = PCA(n_components=12,whiten=True,svd_solver="randomized",random_state=SEED).fit(m)
        x = np.c_[x,np.clip(pca.transform(m),-6.,6.)]
    else:
        scaler,pca = None,None
    if pair_index:
        i,j = np.asarray(pair_index).T
        delta = np.clip(x[i]-x[j],-6.,6.)
        ranked_weight = np.full(len(delta),RANK_WEIGHT*w.sum()/(2.*len(delta)))
        x = np.concatenate((x,delta,-delta))
        y = np.r_[y,np.ones(len(delta),bool),np.zeros(len(delta),bool)]
        w = np.r_[w,ranked_weight,ranked_weight]
    model = LogisticRegression(C=.20,max_iter=1000,random_state=SEED).fit(x,y,sample_weight=w)
    return dict(geometry=scale,motion=scaler,pca=pca,model=model,ranking_pairs=len(pair_index))


def choose(group,models):
    scores = {m:models[m]["model"].predict_proba(encode(group,models[m],m))[:,1] for m in MODELS}
    output = {method:[] for method in METHODS}
    positions = np.split(np.arange(len(group["index"])),np.flatnonzero(np.diff(group["index"]))+1)
    for segment in positions:
        if not len(segment):
            continue
        best = {m:int(segment[np.argmax(scores[m][segment])]) for m in MODELS}
        for method,model,selector in (("static","static","static"),("temporal","temporal","temporal"),
                                      ("t4t5_confirm","t4t5","temporal"),("t4t5_rerank","t4t5","t4t5")):
            i = best[selector]
            output[method].append((int(group["index"][i]),i,float(scores[model][i])))
    return output


def labels_for(group,picks):
    return np.asarray([group["errors"][i] for _,i,_ in picks])


def main():
    assert not set((SILVER,VAL))&set(DEVELOP)
    sources = {Path(__file__),Path(data.__code__.co_filename),Path(budget_cut.__code__.co_filename)}
    groups = {}
    for track in DEVELOP:
        g = data(track,sources)
        groups[track] = g
        selected = set(g["index"][g["target"]].tolist())
        print(track,len(g["index"]),int(g["target"].sum()),len(selected),flush=True)
    folds = {}
    for track in DEVELOP:
        others = [g for name,g in groups.items() if name!=track]
        models = {mode:fit(others,mode) for mode in MODELS}
        folds[track] = choose(groups[track],models)
    total = sum(sum(r["truth"] in ("straight","out_of_range") for r in g["rows"]) for g in groups.values())
    cap = int(FAR_RATE*total)
    cuts, calibration = {},{}
    for method in METHODS:
        all_scores = np.array([s for t in DEVELOP for _,_,s in folds[t][method]])
        all_labels = np.concatenate([labels_for(groups[t],folds[t][method]) for t in DEVELOP])
        cuts[method] = budget_cut(all_scores,all_labels,cap)
        calibration[method] = {t:dict(Counter(groups[t]["errors"][i] for _,i,p in folds[t][method]
                                              if p>=cuts[method])) for t in DEVELOP}
    final = {m:fit(list(groups.values()),m) for m in MODELS}
    checkpoint = OUT/"rankers.joblib"
    joblib.dump(final,checkpoint)
    summary = {t:dict(alternatives=len(g["index"]),correct_alternatives=int(g["target"].sum()),
                      frames_with_correct=len(set(g["index"][g["target"]].tolist())))
               for t,g in groups.items()}
    old.write(OUT/"training.json",dict(status="frozen_before_excluded_replays",development=DEVELOP,
              excluded=[SILVER,VAL],seed=SEED,source_revision=old.revision(),
              false_alert_cap=cap,false_alert_rate=FAR_RATE,cuts=cuts,calibration=calibration,
              ranking_pairs={k:m["ranking_pairs"] for k,m in final.items()},candidate_support=summary,
              checkpoint_sha256=old.digest(checkpoint),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="crossfit_then_freeze"))
    print(cuts,calibration,flush=True)


if __name__ == "__main__":
    main()
