"""Crossfit a confirmation model on candidates selected by an excluded chooser."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
GATE=OUT.parent/"vision-corner-corrected-geometry-gate-20261112-01"
sys.path.insert(0,str(GATE))
from train_gate import (ARCHIVE,DEVELOP,SILVER,VAL,MONZA,old,load_observer,read_development,
    build,pick,split_control,train,fit_phase,fit_confirmers,scores,gate,counts,risk_cut,reference,frames,replay,joblib,
    KINDS,METHODS)
from rank import encode,VALID_CATEGORIES
from selected_head import JointProbability
from sklearn.linear_model import LogisticRegression

MODES=("observed","selected","structured")
FEATURES=("edge","history","motion")


def sampled(group,chosen):
    result={k:group[k][chosen] for k in (*FEATURES,"index","errors","target")}
    result["offers"]=[group["offers"][i] for i in chosen]
    assert len(set(result["index"]))==len(result["index"])
    return result


def target(errors):
    values=np.asarray(errors,dtype="U30").copy()
    values[np.isin(values,("straight_alert","out_of_range_alert"))]="outside_alert"
    return values


def weight_classes(values):
    correct=values=="joint_hit"
    assert correct.any() and (~correct).any()
    weights=np.zeros(len(values))
    weights[correct]=.5/correct.sum()
    reasons=np.unique(values[~correct])
    for reason in reasons:
        use=values==reason
        weights[use]=.5/(len(reasons)*use.sum())
    return weights


def fit_structured(groups,plain):
    models={}
    for kind,model in plain.items():
        columns=[]; categories=[]; weights=[]
        for g in groups:
            use=np.isin(g["errors"],VALID_CATEGORIES)
            current=target(g["errors"][use])
            features={key:g[key][use] for key in FEATURES}
            columns.append(encode(features,model,kind))
            categories.append(current)
            weights.append(weight_classes(current))
        x,y,w=(np.concatenate(values) for values in (columns,categories,weights))
        w *= len(w)/w.sum()
        estimator=LogisticRegression(C=.20,max_iter=1000,random_state=20261112)
        estimator.fit(x,y,sample_weight=w)
        models[kind]=dict(model,model=JointProbability(estimator))
    return models


def fit_selected(groups):
    plain=fit_confirmers(groups,None)
    return dict(selected=plain,structured=fit_structured(groups,plain))


def nested(groups,settings):
    known={track:split_control(g) for track,g in groups.items()}
    cache={}
    def chooser(others):
        key=tuple(others)
        if key not in cache:
            cache[key]=train([known[track] for track in key])
        return cache[key]
    results={mode:{k:[] for k in KINDS} for mode in ("fixed",*MODES)}
    labels=[]
    out_of_fold=[]
    observations={}
    for held,built in groups.items():
        rest=[t for t in groups if t!=held]
        base=[groups[t]["group"] for t in rest]
        phase_model=fit_phase(base)
        picked,raw,phase,indices=pick(built,chooser(rest),phase_model)
        chosen=np.array(list(picked))
        out_of_fold.append(sampled(built["group"],indices))
        labels.append(built["group"]["errors"][indices])
        inner=[]
        for t in rest:
            other=[k for k in rest if k!=t]
            assert held not in other and t not in other
            group=groups[t]
            _,_,_,ix=pick(group,chooser(other),phase_model)
            inner.append(sampled(group["group"],ix))
        variants={"observed":fit_confirmers(base,None),**fit_selected(inner)}
        prediction={"fixed":dict(zip(KINDS,np.asarray(list(raw.values())).T,strict=True))}
        prediction.update({name:scores(built["group"],indices,models) for name,models in variants.items()})
        for name,by_kind in prediction.items():
            for k,prob in by_kind.items():
                results[name][k].append(gate(prob,phase,chosen,settings["phase_cut"]))
        observations[held]=dict(validation=Counter(labels[-1].tolist()),
            inner={t:dict(Counter(g["errors"].tolist())) for t,g in zip(rest,inner,strict=True)})
        print("Nested",held,len(chosen),int(out_of_fold[-1]["target"].sum()),flush=True)
    y=np.concatenate(labels)
    results={mode:{k:np.concatenate(x) for k,x in values.items()} for mode,values in results.items()}
    fixed={k:counts(results["fixed"][k],y,settings["cuts"][m]) for k,m in zip(KINDS,METHODS,strict=True)}
    caps={name:min(c[name] for c in fixed.values()) for name in ("outside","inside")}
    cuts={m:{k:risk_cut(results[m][k],y,caps["outside"],caps["inside"]) for k in KINDS} for m in MODES}
    calibration={m:{k:counts(results[m][k],y,cuts[m][k]) for k in KINDS} for m in MODES}
    return cuts,calibration,caps,fixed,out_of_fold,observations


def main():
    model,settings,sources=load_observer()
    sources.update((Path(__file__),OUT/"selected_head.py",
                    Path(read_development.__code__.co_filename),Path(split_control.__code__.co_filename),
                    Path(encode.__code__.co_filename),Path(train.__code__.co_filename),Path(fit_phase.__code__.co_filename)))
    groups={}
    for track in DEVELOP:
        sensor,truth,_=read_development(track,sources,settings)
        groups[track]=build(sensor,settings,truth)
    cuts,development,caps,fixed,selected,observations=nested(groups,settings)
    prior=old.read(GATE/"training.json")
    assert caps==prior["caps"] and cuts["observed"]==prior["cuts"]["observed"]
    confirmers={"observed":fit_confirmers([g["group"] for g in groups.values()],None),
                **fit_selected(selected)}
    output=OUT/"confirmers.joblib"
    joblib.dump(confirmers,output)
    frozen=OUT/"training.json"
    assert not any(t in str(p) for p in sources for t in (SILVER,VAL,MONZA))
    training=dict(status="frozen_before_replay",development=DEVELOP,excluded=[SILVER,VAL,MONZA],
              cuts=cuts,development_metrics=development,reference_metrics=fixed,caps=caps,
              folds=observations,seed=20261112,source_revision=old.revision(),checkpoint_sha256=old.digest(output),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="nested_excluded_choice_calibration")
    old.write(frozen,training)
    print("Frozen",cuts,development,flush=True)
    results={}
    for track,folder in ((SILVER,"silverstone"),(VAL,"vallelunga"),(MONZA,"monza")):
        root=ARCHIVE/folder
        sensor=root/"sensors.npz"
        assert old.digest(sensor)==old.read(root/"result.json")["inputs_sha256"]
        prior_output=root/"predictions/forecasts.jsonl"
        original=frames(prior_output,old.read(prior_output.with_name("result.json"))["forecasts_sha256"])
        truth,ref_path=reference(track)
        rows,metrics=replay(sensor,truth,original,model,settings,confirmers,training)
        trace=OUT/f"{track}-frames.jsonl"
        old.jsonl(trace,rows)
        results[track]=dict(frames=len(rows),rows_sha256=old.digest(trace),metrics=metrics,
                            source_sha256={str(p):old.digest(p) for p in (sensor,prior_output,ref_path,frozen,output)})
        print(track,metrics,flush=True)
    old.write(OUT/"result.json",dict(status="scored",training_sha256=old.digest(frozen),
             tracks=results,termination="selected_candidate_confirmation_replayed"))


if __name__=="__main__":
    main()
