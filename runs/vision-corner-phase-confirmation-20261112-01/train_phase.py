"""Fit a shared entry-phase detector and choose its operating rules by circuit folds."""
from collections import Counter
from pathlib import Path
import math
import numpy as np

from phase_features import OUT, PARENT, DEVELOP, SILVER, VAL, group, old
from rank import MODELS, METHODS, fit, choose, labels_for, joblib
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 20261112
FALSE_RATE = .05
VETO_KEEP = .90
OUTSIDE = ("straight_alert","out_of_range_alert")


def fit_phase(groups):
    features, labels, weights = [],[],[]
    for g in groups:
        y = g["phase_y"]
        keep = (y>=0)&np.isfinite(g["phase_x"]).all(1)
        features.append(g["phase_x"][keep])
        y = y[keep]
        labels.append(y)
        freq = np.bincount(y,minlength=3)
        weights.append(1./freq[y])
    x,y,w = (np.concatenate(v) for v in (features,labels,weights))
    assert set(y)=={0,1,2}
    w *= len(w)/w.sum()
    return HistGradientBoostingClassifier(max_iter=75,max_leaf_nodes=12,min_samples_leaf=30,
             learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=SEED).fit(x,y,sample_weight=w)


def phase_scores(g,model):
    assert model.classes_.tolist()==[0,1,2]
    valid = np.isfinite(g["phase_x"]).all(1)
    output = np.full((len(valid),3),np.nan)
    output[valid] = model.predict_proba(g["phase_x"][valid])
    return output


def counts(accepted,labels):
    return dict(Counter(labels[accepted].tolist()))


def veto_cut(scores,baseline_cuts):
    choices = sorted({1.,*[float(q) for method in ("temporal","t4t5_confirm")
                         for q in scores[method]["corner"]]})
    required = {}
    for method in ("temporal","t4t5_confirm"):
        a = scores[method]
        required[method] = math.ceil(VETO_KEEP*int(((a["score"]>=baseline_cuts[method])
                                                   &(a["category"]=="joint_hit")).sum()))
    for limit in choices:
        if all(int(((scores[m]["score"]>=baseline_cuts[m])&(scores[m]["corner"]<=limit)
                    &(scores[m]["category"]=="joint_hit")).sum())>=required[m] for m in required):
            return limit,required
    raise ValueError("No phase-veto operating point")


def risk_cut(p,labels,max_outside,max_inside):
    finite = np.isfinite(p)
    assert finite.all()
    outside = np.isin(labels,OUTSIDE)
    inside = labels=="in_corner_alert"
    correct = labels=="joint_hit"
    candidates = np.r_[np.nextafter(p.max(),np.inf),np.unique(p)]
    options = []
    for cut in candidates:
        accepted = p>=cut
        a,b = int((accepted&outside).sum()),int((accepted&inside).sum())
        if a<=max_outside and b<=max_inside:
            options.append((int((accepted&correct).sum()),-a-b,-a,-b,float(cut)))
    return max(options)[-1]


def fold_rows(g,picks,phase):
    output = {}
    for m,options in picks.items():
        frames = np.asarray([f for f,_,_ in options],dtype=int)
        corner = phase[frames,2]
        assert np.isfinite(corner).all()
        output[m] = dict(score=np.asarray([p for _,_,p in options]),corner=corner,
                         category=labels_for(g,options))
    return output


def main():
    sources = {Path(__file__),Path(group.__code__.co_filename),Path(fit.__code__.co_filename),
               Path(choose.__code__.co_filename),PARENT/"training.json"}
    previous = old.read(PARENT/"training.json")
    assert previous["development"]==list(DEVELOP)
    groups = {t:group(t,sources) for t in DEVELOP}
    oof, phase_report = {},{}
    for track in DEVELOP:
        others = [g for t,g in groups.items() if t!=track]
        rankers = {m:fit(others,m) for m in MODELS}
        g = groups[track]
        phase = phase_scores(g,fit_phase(others))
        valid = (g["phase_y"]>=0)&np.isfinite(phase).all(1)
        cm = np.zeros((3,3),dtype=int)
        np.add.at(cm,(g["phase_y"][valid],phase[valid].argmax(1)),1)
        phase_report[track] = cm.tolist()
        oof[track] = fold_rows(g,choose(g,rankers),phase)
        print(track,cm.tolist(),flush=True)
    joined = {m:{k:np.concatenate([oof[t][m][k] for t in DEVELOP])
                 for k in ("score","corner","category")} for m in METHODS}
    counts_outside = sum(sum(r["truth"] in ("straight","out_of_range") for r in g["rows"]) for g in groups.values())
    counts_inside = sum(sum(r["truth"]=="in_corner" for r in g["rows"]) for g in groups.values())
    caps = (int(FALSE_RATE*counts_outside),int(FALSE_RATE*counts_inside))
    baseline_cuts = previous["cuts"]
    limit,required = veto_cut(joined,baseline_cuts)
    calibration,cuts = {},{}
    for m,g in joined.items():
        p,q,y = (g[k] for k in ("score","corner","category"))
        base_cut = risk_cut(p,y,*caps)
        guarded = p*(1-q)
        guard_cut = risk_cut(guarded,y,*caps)
        cuts[m] = dict(original=baseline_cuts[m],risk=base_cut,phase_risk=guard_cut)
        calibration[m] = dict(original=counts(p>=baseline_cuts[m],y),
                              veto=counts((p>=baseline_cuts[m])&(q<=limit),y),
                              risk=counts(p>=base_cut,y),phase_risk=counts(guarded>=guard_cut,y))
    checkpoint = OUT/"phase.joblib"
    joblib.dump(fit_phase(list(groups.values())),checkpoint)
    old.write(OUT/"training.json",dict(status="frozen_before_two_replays",development=DEVELOP,excluded=[SILVER,VAL],
             seed=SEED,source_revision=old.revision(),cuts=cuts,veto_corner_probability_max=limit,
             veto_required_development_hits=required,development_false_caps=dict(outside=caps[0],in_corner=caps[1]),
             calibration=calibration,phase_confusion_by_circuit=phase_report,
             phase_classes=["outside","approach","in_corner"],checkpoint_sha256=old.digest(checkpoint),
             source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="shared_phase_and_limits_frozen"))
    print("veto",limit,"caps",caps,calibration,flush=True)


if __name__=="__main__":
    main()
