"""Crossfit two confirmation heads with and without physical geometry supervision."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from gate_inputs import OUT,ROOT,DEVELOP,SILVER,VAL,old,load_observer,read_development,corrected,build,pick,scores
from rank import fit,joblib
from listwise import train
from train_phase import fit_phase,risk_cut
sys.path.insert(0,str(OUT.parent/"vision-corner-paired-risk-20261112-01"))
from paired_risk import ARCHIVE,MONZA,reference,counts,METHODS,frames

MODES=("observed","augmented")
KINDS=("temporal","t4t5")


def fit_confirmers(groups,augmented):
    data=list(groups)
    if augmented:
        data.extend(augmented)
    return {m:fit(data,m) for m in KINDS}


def gate(raw,phase,frame_index,limit):
    return np.where(np.isfinite(raw)&np.isfinite(phase[frame_index])&(phase[frame_index]<=limit),raw,-1.)


def split_control(built):
    group=built["group"]
    original=group["original"]
    return {k:([row for row,m in zip(v,original,strict=True) if m] if k=="offers" else v[original])
            if k in ("edge","history","motion","index","offers","errors","target") else v
            for k,v in group.items()}


def prepare(settings,sources):
    original,physical,examples={},{},{}
    for track in DEVELOP:
        data,truth,proof=read_development(track,sources,settings)
        original[track]=build(data,settings,truth)
        auxiliary=corrected(track,data,proof,sources)
        row=original[track]["group"]
        examples[track]=dict(observed=dict(candidates=len(row["index"]),joint=int(row["target"].sum())))
        if auxiliary is not None:
            sensors,n=auxiliary
            group=build(sensors,settings,truth)["group"]
            physical[track]=group
            examples[track]["corrected"]=dict(points=n,candidates=len(group["index"]),joint=int(group["target"].sum()))
        print("Examples",track,examples[track],flush=True)
    return original,physical,examples


def calibrate(original,physical,settings):
    values={name:{m:[] for m in KINDS} for name in ("fixed",*MODES)}
    labels=[]
    for held,built in original.items():
        rest=[t for t in original if t!=held]
        base=[original[t]["group"] for t in rest]
        chooser=train([split_control(original[t]) for t in rest])
        phase_model=fit_phase(base)
        offers,raw,q,index=pick(built,chooser,phase_model)
        assert len(index)==len(offers)
        chosen=np.array(list(offers))
        categories=built["group"]["errors"][index]
        labels.append(categories)
        comparison={
            "fixed":dict(zip(KINDS,np.asarray(list(raw.values())).T,strict=True)),
            "observed":scores(built["group"],index,fit_confirmers(base,None)),
            "augmented":scores(built["group"],index,fit_confirmers(base,[physical[t] for t in rest if t in physical])),
        }
        for name,mode in comparison.items():
            for m,p in mode.items():
                values[name][m].append(gate(p,q,chosen,settings["phase_cut"]))
        print("Crossfit",held,len(categories),flush=True)
    y=np.concatenate(labels)
    values={name:{k:np.concatenate(v) for k,v in group.items()} for name,group in values.items()}
    fixed={m:counts(values["fixed"][m],y,settings["cuts"][legacy]) for m,legacy in zip(KINDS,METHODS,strict=True)}
    caps={key:min(v[key] for v in fixed.values()) for key in ("outside","inside")}
    cuts,metrics={},{}
    for mode in MODES:
        cuts[mode]={m:risk_cut(values[mode][m],y,caps["outside"],caps["inside"]) for m in KINDS}
        metrics[mode]={m:counts(values[mode][m],y,cuts[mode][m]) for m in KINDS}
    return cuts,metrics,caps,fixed


def replay(path,truth,archive_forecasts,models,settings,confirmers,training):
    with np.load(path,allow_pickle=False) as data:
        sensors={k:data[k] for k in data.files}
    built=build(sensors,settings)
    offers,raw,q,index=pick(built,models["rank"],models["phase"])
    chosen=np.array(list(offers))
    assert np.array_equal(built["group"]["index"][index],chosen)
    comparison={"fixed":dict(zip(KINDS,np.asarray(list(raw.values())).T,strict=True)),
                **{name:scores(built["group"],index,m) for name,m in confirmers.items()}}
    offered={i:k for k,i in enumerate(chosen)}
    for label,model in comparison.items():
        for m in KINDS:
            model[m]=gate(model[m],q,chosen,settings["phase_cut"])
    for name,method in zip(KINDS,METHODS,strict=True):
        cut=settings["cuts"][method]
        if archive_forecasts is not None:
            for i,row in enumerate(archive_forecasts):
                assert (row["run"],row["frame"])==(built["rows"][i]["run"],built["rows"][i]["frame"])
                actual=row["forecasts"][method]
                accepted=i in offered and comparison["fixed"][name][offered[i]]>=cut
                assert accepted==(actual["state"] in ("left","right"))
                if accepted:
                    assert offers[i][1]==actual["distance_m"] and actual["state"]==("left","right")[offers[i][0]-1]
    table={(r["run"],r["frame"]):r for r in truth}
    labels={}
    rows=[]
    metrics={}
    for i,row in enumerate(built["rows"]):
        key=row["run"],row["frame"]
        if key not in table:
            continue
        ref=table[key]
        decisions={}
        for mode,ps in comparison.items():
            for m,method in zip(KINDS,METHODS,strict=True):
                cut=settings["cuts"][method] if mode=="fixed" else training["cuts"][mode][m]
                j=offered.get(i)
                p=ps[m][j] if j is not None else -1.
                prediction=offers[i] if p>=cut else None
                category=old.category(ref,prediction)
                counter=metrics.setdefault(f"{mode}_{m}",Counter())
                counter[category]+=1
                if category=="joint_hit":
                    counter[f"joint_{ref['label']}"]+=1
                decisions[f"{mode}_{m}"]=dict(category=category,score=float(p),accepted=prediction is not None)
        rows.append(dict(run=key[0],frame=key[1],truth=ref["truth"],label=ref["label"],
                         candidate=offers.get(i),decisions=decisions))
    assert len(rows)==len(truth)
    return rows,{k:dict(v) for k,v in metrics.items()}


def main():
    models,settings,sources=load_observer()
    sources.update((Path(__file__),Path(read_development.__code__.co_filename),Path(fit.__code__.co_filename),
                    Path(train.__code__.co_filename),Path(fit_phase.__code__.co_filename)))
    original,physical,examples=prepare(settings,sources)
    cuts,dev,caps,fixed=calibrate(original,physical,settings)
    base=[g["group"] for g in original.values()]
    confirmers={"observed":fit_confirmers(base,None),"augmented":fit_confirmers(base,list(physical.values()))}
    checkpoint=OUT/"confirmers.joblib"
    joblib.dump(confirmers,checkpoint)
    frozen=OUT/"training.json"
    assert not any(t in str(p) for t in (SILVER,VAL,MONZA) for p in sources)
    training=dict(status="frozen_before_replay",development=DEVELOP,physical_development=list(physical),
              excluded=[SILVER,VAL,MONZA],cuts=cuts,development_metrics=dev,reference_metrics=fixed,caps=caps,
              samples=examples,seed=20261112,source_revision=old.revision(),checkpoint_sha256=old.digest(checkpoint),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="observed_only_crossfit_calibration")
    old.write(frozen,training)
    print("Frozen",cuts,dev,caps,flush=True)
    reports={}
    for t,folder in ((SILVER,"silverstone"),(VAL,"vallelunga"),(MONZA,"monza")):
        base_path=ARCHIVE/folder
        path=base_path/"sensors.npz"
        receipt=base_path/"result.json"
        assert old.digest(path)==old.read(receipt)["inputs_sha256"]
        output=base_path/"predictions/forecasts.jsonl"
        original_prediction=frames(output,old.read(output.with_name("result.json"))["forecasts_sha256"])
        truth,ref_path=reference(t)
        rows,metrics=replay(path,truth,original_prediction,models,settings,confirmers,training)
        witness=OUT/f"{t}-frames.jsonl"
        old.jsonl(witness,rows)
        sources=(path,receipt,output,ref_path,frozen,checkpoint)
        reports[t]=dict(frames=len(rows),rows_sha256=old.digest(witness),metrics=metrics,
                        source_sha256={str(p):old.digest(p) for p in sources})
        print(t,metrics,flush=True)
    old.write(OUT/"result.json",dict(status="frozen_comparison_complete",training_sha256=old.digest(frozen),
              tracks=reports,termination="real_geometry_replay_with_common_candidates"))


if __name__=="__main__":
    main()
