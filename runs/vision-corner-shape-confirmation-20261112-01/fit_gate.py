"""Fit paired candidate confirmation on four development circuits and cross-fit its cuts."""
from pathlib import Path
from collections import Counter

from features import OUT, DEVELOP, SILVER, data, old

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



SEED = 20261112
FAR_RATE = .05
MODES = ("edge","t4t5")
OUTSIDE = ("straight_alert","out_of_range_alert")
VALID_CATEGORIES = ("joint_hit","wrong_direction","distance_error","straight_alert",
                    "out_of_range_alert","in_corner_alert")


def fit(groups,mode):
    items = []
    for g in groups:
        mask = np.isin(g["errors"],VALID_CATEGORIES)
        target = g["target"][mask]
        weights = np.empty(len(target),dtype=float)
        for status in (False,True):
            subset = target==status
            weights[subset] = .5/max(1,subset.sum())
        items.append((g["edge"][mask],g["motion"][mask],target,weights))
    edge, motion, target, weights = (np.concatenate(v) for v in zip(*items,strict=True))
    assert set(target)=={False,True}
    weights *= len(weights)/weights.sum()
    e = StandardScaler().fit(edge,sample_weight=weights)
    x = np.clip(e.transform(edge),-6.,6.)
    if mode=="t4t5":
        m = StandardScaler().fit(motion)
        standardized = np.clip(m.transform(motion),-6.,6.)
        pca = PCA(n_components=min(12,len(motion)-1,motion.shape[1]),
                  svd_solver="randomized",random_state=SEED,whiten=True).fit(standardized)
        x = np.c_[x,np.clip(pca.transform(standardized),-6.,6.)]
    else:
        m,pca = None,None
    reg = LogisticRegression(C=.20,max_iter=1000,random_state=SEED)
    reg.fit(x,target,sample_weight=weights)
    return dict(edge=e,motion=m,pca=pca,regressor=reg)


def scores(group,model):
    x = np.clip(model["edge"].transform(group["edge"]),-6.,6.)
    if model["motion"] is not None:
        motion = np.clip(model["motion"].transform(group["motion"]),-6.,6.)
        x = np.c_[x,np.clip(model["pca"].transform(motion),-6.,6.)]
    return model["regressor"].predict_proba(x)[:,1]


def budget_cut(probability,errors,maximum):
    far = np.isin(errors,OUTSIDE)
    joint = errors=="joint_hit"
    choices = np.r_[np.nextafter(probability.max(),np.inf),np.unique(probability)]
    ranked = []
    for cut in choices:
        accepted = probability>=cut
        count = int((accepted&far).sum())
        if count<=maximum:
            ranked.append(((int((accepted&joint).sum()),-count,-int(accepted.sum()),float(cut)),float(cut)))
    return max(ranked)[1]


def grouped_counts(group,probability,cut):
    return dict(Counter(group["errors"][probability>=cut].tolist()))


def main():
    assert SILVER not in DEVELOP
    sources = {Path(__file__),Path(data.__code__.co_filename)}
    groups = {}
    for track in DEVELOP:
        g = data(track,sources)
        groups[track] = g
        print(track,len(g["rows"]),len(g["index"]),dict(Counter(g["errors"].tolist())),flush=True)
    folds = {mode:{} for mode in MODES}
    for mode in MODES:
        for track in DEVELOP:
            others = [g for name,g in groups.items() if name!=track]
            folds[mode][track] = scores(groups[track],fit(others,mode))
    far_total = sum(sum(r["truth"] in ("straight","out_of_range") for r in g["rows"]) for g in groups.values())
    cap = int(FAR_RATE*far_total)
    errors = np.concatenate([g["errors"] for g in groups.values()])
    thresholds, models, calibration = {},{},{}
    for mode in MODES:
        crossfit = np.concatenate([folds[mode][t] for t in DEVELOP])
        thresholds[mode] = budget_cut(crossfit,errors,cap)
        calibration[mode] = {t:grouped_counts(g,folds[mode][t],thresholds[mode]) for t,g in groups.items()}
        models[mode] = fit(list(groups.values()),mode)
    path = OUT/"confirmation.joblib"
    joblib.dump(models,path)
    receipt = dict(status="frozen_before_silverstone",seed=SEED,source_revision=old.revision(),
                   development=list(DEVELOP),excluded=[SILVER],crossfit="one circuit held out per fold",
                   threshold_by_model=thresholds,development_far_frames=far_total,
                   outside_budget=cap,development_outside_rate=FAR_RATE,calibration=calibration,
                   candidate_counts={t:len(g["index"]) for t,g in groups.items()},
                   availability="both models require the same valid candidate and T4/T5 feature bundle",
                   checkpoint_sha256=old.digest(path),
                   source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                   termination="paired_models_and_crossfit_cuts_frozen")
    old.write(OUT/"training.json",receipt)
    print(thresholds,calibration,flush=True)


if __name__ == "__main__":
    main()
