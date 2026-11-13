"""Test whether assigning background rejection to the presence head helps transfer."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from heads import (CALIBRATION_TRACKS, CUTS, OUT, SEED, SILVER, TRACKS,
                   examples, feasible, load, load_silver, old_directions, render, review)
from train_vision_teacher_line import digest, write


def fit_head(x, y, w):
    w = w * len(w) / w.sum()
    scale = StandardScaler().fit(x,sample_weight=w)
    head = LogisticRegression(C=.08,random_state=SEED,max_iter=600)
    head.fit(scale.transform(x).clip(-8.,8.), y,sample_weight=w)
    return scale,head


def classify(model, x):
    scale,head = model
    return [head.predict_proba(scale.transform(x[:,i]).clip(-8.,8.)) for i in range(2)]


def main():
    hashes = {Path(__file__),OUT/"heads.py"}
    tracks = (*TRACKS,SILVER)
    groups = {t:load(t,hashes) for t in TRACKS}
    groups[SILVER] = load_silver(hashes)
    samples = {t:examples(groups[t]) for t in CALIBRATION_TRACKS}
    baseline = {t:review(g,render(g,old_directions(g["scores"]),g["presence"],g["presence_threshold"]))
                for t,g in groups.items()}

    @lru_cache
    def predict(track, training):
        assert track not in training and all(t in CALIBRATION_TRACKS for t in training)
        x,y,w = (np.concatenate(parts) for parts in zip(*(samples[t] for t in training),strict=True))
        near = y!=0
        g=groups[track]
        p,q = classify(fit_head(x,near,w),g["x"])
        left,right = classify(fit_head(x[near],y[near],w[near]),g["x"])
        scores = .5*(left+right[:,::-1])
        sides = np.where(scores[:,0]>scores[:,1],1,np.where(scores[:,1]>scores[:,0],2,0))
        return sides,.5*(p[:,1]+q[:,1])

    def score(track, training, cut):
        return review(groups[track],render(groups[track],*predict(track,training),cut))

    result={}
    for track in tracks:
        training=tuple(t for t in CALIBRATION_TRACKS if t!=track)
        nested={t:tuple(k for k in training if k!=t) for t in training}
        ranked=[]
        for cut in CUTS:
            scores={t:score(t,peers,cut) for t,peers in nested.items()}
            if not all(feasible(baseline[t],s) for t,s in scores.items()):
                continue
            gains=tuple(sum(scores[t]["metrics"]["candidate"][f]-baseline[t]["metrics"]["candidate"][f]
                            for t in training) for f in ("left","joint","right"))
            stable=sum(scores[t]["stable_physical"]-baseline[t]["stable_physical"] for t in training)
            if any(gains) or stable:
                ranked.append(((stable,*gains,cut),cut))
        picked=max(ranked)[1] if ranked else None
        result[track]=dict(training=training,calibration=nested,cutoff=picked,baseline=baseline[track],
            selected=score(track,training,picked) if picked is not None else None,
            sweep={str(c):score(track,training,c) for c in CUTS})
        print(track,picked,flush=True)
    write(OUT/"conditional-result.json",dict(status="evaluated",seed=SEED,tracks=result,
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
