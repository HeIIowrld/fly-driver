"""Audit left/right onset failures and compare all plausible boundary candidates."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent/"vision-corner-onset-through-line-20261113-01"))
from dense_onset import (CLI, EVAL, DEVELOP, METHODS, TURNS, RISK, SEED, choose_cut, frames, label,
                         load_observer, old, read_development, reference, score_archive, turning)
from gate_inputs import SensorArrays, build
from replay_corner_boundary_states import observe, phase_scores
from rank import fit, encode

KINDS = ("temporal","t4t5")
MODES = ("fixed_alternatives","refit_alternatives")


def models_for(groups):
    return {kind:fit(groups,kind) for kind in KINDS}


def probabilities(g, models):
    result=[]
    complete = np.isfinite(g["motion"]).all(1)
    for kind in KINDS:
        use = complete if kind=="t4t5" else np.ones(len(complete),bool)
        p = np.full(len(use),-1.)
        if use.any():
            inputs={k:g[k][use] for k in ("edge","history","motion")}
            p[use]=models[kind]["model"].predict_proba(encode(inputs,models[kind],kind))[:,1]
        result.append(p)
    return np.column_stack(result)


def choices(built, base, models, observer, settings):
    g=built["group"]
    probabilities_by_option=probabilities(g,models)
    interior=phase_scores(g,observer["phase"])[:,2]
    best={}
    for j,i in enumerate(g["index"]):
        if turning(base[i]) or not interior[i]<=settings["phase_cut"]:
            continue
        if i not in best or probabilities_by_option[j,0]>probabilities_by_option[best[i],0]:
            best[i]=j
    return best, probabilities_by_option


def audit(built, base, truth, observer, settings):
    group=built["group"]
    available=defaultdict(list)
    for i,offer in zip(group["index"],group["offers"],strict=True):
        available[i].append(offer)
    rows={(a["run"],a["frame"]):i for i,a in enumerate(base)}
    q=phase_scores(group,observer["phase"])[:,2]
    result=[]
    for ref in truth:
        if ref["truth"]!="approach":
            continue
        i=rows[ref["run"],ref["frame"]]
        direction=int(ref["label"])
        target=ref["truth_endpoints_m"]
        options=available[i]
        selected=base[i]["candidate"]
        same=[a for a in options if a[0]==direction]
        close=[a for a in same if label(ref,*a)=="hit_"+ref["label"]]
        forecast=base[i]["forecasts"]["t4t5_confirm"]
        def reason():
            if not same:
                return "no_same_direction_candidate"
            if not close:
                return "no_candidate_within_5m"
            if selected is None or label(ref,*selected)!="hit_"+ref["label"]:
                return "selection_lost_matching_candidate"
            if TURNS.get(forecast["state"])==direction:
                return "joint_hit"
            if q[i]>settings["phase_cut"]:
                return "in_corner_veto"
            if not np.isfinite(built["sensors"]["t4t5"][i]).all():
                return "motion_missing"
            return "confirmation_cut"
        result.append(dict(run=ref["run"],frame=ref["frame"],side=direction,truth_m=target,selected=selected,
                           candidate_pool=options,matching=close,reason=reason(),forecast=forecast,
                           in_corner_score=float(q[i]) if np.isfinite(q[i]) else None))
    return result


def calibrate(groups, observer, settings):
    oof={mode:[[],[]] for mode in MODES}
    frozen={k:observer["rank"][k]["confirmer"] for k in KINDS}
    for held,(built,base,truth) in groups.items():
        fresh=models_for([b["group"] for k,(b,_,_) in groups.items() if k!=held])
        for mode,model in zip(MODES,(frozen,fresh),strict=True):
            picked,p=choices(built,base,model,observer,settings)
            cats=np.array([label(truth[i],*built["group"]["offers"][j]) for i,j in picked.items()])
            for m in range(2):
                oof[mode][m].append((p[list(picked.values()),m],cats))
        print("Calibrated",held,flush=True)
    cap=dict(outside=8,inside=5,wrong_direction=2,distance_error=2)
    cuts={name:[choose_cut(entries,cap) for entries in by_method] for name,by_method in oof.items()}
    def utility(name):
        counts=sum((Counter(r[1]) for r in cuts[name]),Counter())
        return min(counts["hit_1"],counts["hit_2"]),counts["hit_1"]+counts["hit_2"],-sum(counts[k] for k in RISK)
    chosen=max(MODES,key=utility)
    return cuts,chosen,cap,dict(zip(MODES,(frozen,models_for([a[0]["group"] for a in groups.values()])),strict=True))


def apply(built, base, picked, score, cuts):
    final=list(base)
    for i,j in picked.items():
        side,distance=built["group"]["offers"][j]
        accept=[score[j,k]>=cut[0] for k,cut in enumerate(cuts)]
        if any(accept):
            forecasts=dict(base[i]["forecasts"])
            for method,ok in zip(METHODS,accept,strict=True):
                if ok:
                    forecasts[method]=dict(state=("left","right")[side-1],distance_m=distance)
            final[i]=dict(base[i],candidate=[side,distance],forecasts=forecasts)
    return final


def run():
    observer,settings,sources=load_observer()
    sources.update((Path(__file__),Path(fit.__code__.co_filename),Path(build.__code__.co_filename),
                    Path(read_development.__code__.co_filename),Path(choose_cut.__code__.co_filename)))
    groups={}
    for track in DEVELOP:
        sensors,truth,_=read_development(track,sources,settings)
        data=SensorArrays(sensors)
        groups[track]=(build(data,settings,truth),observe(data,observer,settings)[0],truth)
    cuts,chosen,cap,models=calibrate(groups,observer,settings)
    model_file=OUT/"alternative-confirmers.joblib"
    joblib.dump(models,model_file)
    frozen=OUT/"training.json"
    old.write(frozen,dict(development=list(groups),excluded=[t for t,_ in EVAL],chosen=chosen,cuts=cuts,
              risk=cap,checkpoint_sha256=old.digest(model_file),seed=SEED,source_revision=old.revision(),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="frozen_before_evaluation"))
    print(chosen,cuts,flush=True)
    results={}
    for track,name in EVAL:
        root=CLI/("vallelunga-retry" if name=="vallelunga" else name)
        receipt=root/"result.json"
        input_file=root/"recovered-sensors.npz"
        assert old.digest(input_file)==old.read(receipt)["recovered_input_sha256"]
        with np.load(input_file,allow_pickle=False) as data:
            sensors=SensorArrays({k:data[k] for k in data.files})
        built=build(sensors,settings)
        forecast_file=root/"predictions/forecasts.jsonl"
        base=frames(forecast_file,old.read(forecast_file.with_name("result.json"))["forecasts_sha256"])
        assert [(r["run"],r["frame"]) for r in base]==[(r["run"],r["frame"]) for r in built["rows"]]
        truth,truth_path=reference(track)
        detail=audit(built,base,truth,observer,settings)
        issues={side:Counter(r["reason"] for r in detail if r["side"]==side) for side in (1,2)}
        audit_path=OUT/f"{track}-audit.jsonl"
        old.jsonl(audit_path,detail)
        metrics={}
        for mode in ("baseline",*MODES):
            if mode=="baseline":
                revised=base
            else:
                picked,score=choices(built,base,models[mode],observer,settings)
                revised=apply(built,base,picked,score,cuts[mode])
            assert all(a==b for a,b in zip(base,revised,strict=True) if turning(a))
            index={(r["run"],r["frame"]):r for r in revised}
            method_scores={}
            for method in METHODS:
                counts=score_archive(truth,index,method)
                errors=defaultdict(list)
                for t in truth:
                    if t["truth"]!="approach":
                        continue
                    f=index[t["run"],t["frame"]]["forecasts"][method]
                    if TURNS.get(f["state"])==int(t["label"]):
                        errors[t["label"]].append(max(abs(f["distance_m"]-d) for d in t["truth_endpoints_m"]))
                method_scores[method]=dict(counts=counts,distance={side:dict(n=len(v),mae_m=float(np.mean(v))) for side,v in errors.items()})
            witness=OUT/f"{track}-{mode}.jsonl"
            old.jsonl(witness,revised)
            metrics[mode]=dict(methods=method_scores,forecast_sha256=old.digest(witness))
        results[track]=dict(issues=issues,audit_sha256=old.digest(audit_path),metrics=metrics,
            source_sha256={str(p):old.digest(p) for p in (receipt,input_file,forecast_file,truth_path)})
        print(track,issues,{k:{m:v["counts"] for m,v in s["methods"].items()} for k,s in metrics.items()},flush=True)
    old.write(OUT/"result.json",dict(chosen=chosen,tracks=results,training_sha256=old.digest(frozen),termination="archive_end"))


if __name__=="__main__":
    run()
