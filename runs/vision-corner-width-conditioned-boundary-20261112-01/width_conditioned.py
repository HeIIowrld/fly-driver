"""Recover missing side coordinates using measured road-width context and temporal donors."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

OUT=Path(__file__).resolve().parent
POLICY=OUT.parent/"vision-corner-boundary-policy-20261112-01"
sys.path.insert(0,str(POLICY))
from calibrate_policy import (ARCHIVE, DEVELOP, DISTANCE_LIMIT_M, EVALUATION, EXTRA, METHODS, POLICIES, PREV, SEED, SIGNS,
    SILVER, SensorArrays, boundary_scores, combine, examples, features, frames, joblib, load_labels, load_observer,
    observe, old, patch, read_development, reference, row_map, selected, forecast_errors)

LIMITS=(.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1.01)
SOURCES=(1,11,12,13,14)


def local_width(raw):
    width=raw[:,:,1]-raw[:,:,0]
    valid=(width>3.)&(width<20.)
    support=np.full(width.shape,np.nan)
    mad=np.full(width.shape,np.nan)
    count=np.zeros(width.shape)
    for j in range(width.shape[1]):
        start,end=max(0,j-3),min(width.shape[1],j+4)
        good=valid[:,start:end]
        n=good.sum(1)
        have=n>=2
        sample=np.where(good[have],width[have,start:end],np.nan)
        value=np.nanmedian(sample,axis=1)
        support[have,j]=value
        mad[have,j]=np.nanmedian(abs(sample-value[:,None]),axis=1)
        count[:,j]=n
    return support,mad,count


def bank(sensors,settings):
    f,_=features(sensors,settings)
    width,mad,count=local_width(sensors["sections"])
    opposite=f[...,6]
    signed_width=width[:,:,None]-opposite
    donor=np.concatenate((f[...,SOURCES],signed_width[...,None]),axis=-1)
    available=~np.isfinite(f[...,0]) & np.isfinite(donor).any(axis=-1)
    pos=np.argwhere(available)
    context=f[available]
    candidates=donor[available]
    cell,source=np.nonzero(np.isfinite(candidates))
    j=tuple(pos.T)
    row,station,_=pos.T
    expected=width[row,station]
    error=mad[row,station]
    n=count[row,station]
    value=candidates[cell,source]
    votes=(abs(value[:,None]-candidates[cell,:5])<=.75).sum(axis=1)
    counts=np.isfinite(candidates[cell]).sum(axis=1)
    span=abs(value[:,None]-context[cell][:,[2,3,4,5]])
    x=np.column_stack((context[cell,1:],source,value,counts,votes,span,
                       expected[cell],error[cell],n[cell],value+opposite[j][cell]-expected[cell]))
    return dict(positions=pos,cell=cell,source=source,value=value,x=x,counts=counts)


def fit(groups):
    x,y,w=[],[],[]
    for b,edge,visible in groups:
        j=tuple(b["positions"].T)
        cell=b["cell"]
        use=visible[j][cell]
        errors=abs(b["value"]-(edge*SIGNS)[j][cell])
        x.append(b["x"][use])
        y.append(errors[use]<=DISTANCE_LIMIT_M)
        weight=1./b["counts"][use]
        w.append(weight/weight.sum())
    a,b,c=map(np.concatenate,(x,y,w))
    model=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,min_samples_leaf=40,
             l2_regularization=2.,random_state=SEED)
    return model.fit(a,b,sample_weight=c/c.mean())


def choose(bank,model):
    p=model.predict_proba(bank["x"])[:,1]
    cell,origin=bank["cell"],bank["source"]
    order=np.lexsort((origin,-p,cell))
    _,start=np.unique(cell[order],return_index=True)
    best=order[start]
    assert np.array_equal(cell[best],np.arange(len(bank["positions"])))
    return p[best],bank["value"][best],origin[best]


def apply(raw,bank,picked,cut):
    j=tuple(bank["positions"].T)
    p,value,_=picked
    signed=np.full_like(raw,np.nan)
    signed[j]=value
    take=np.zeros(raw.shape,bool)
    take[j]=p>=cut
    return combine(raw,signed,take,np.zeros(raw.shape,bool))


def quality(raw,new,edge,visible):
    sample=~np.isfinite(raw)&np.isfinite(new)&visible
    error=abs(new[sample]-edge[sample])
    return dict(accepted=int(sample.sum()),correct=int((error<=DISTANCE_LIMIT_M).sum()),error_m=float(error.sum()))


def development(data):
    fits={}
    for held,item in data.items():
        model=fit([v["training"] for t,v in data.items() if t!=held])
        raw=item["sensors"]["sections"]
        bank,edge,visible=item["training"]
        picked=choose(bank,model)
        fits[held]={c:quality(raw,apply(raw,bank,picked,c),edge,visible) for c in LIMITS}
    summed={}
    for c in LIMITS:
        summary=Counter()
        for track in fits.values():
            summary.update(track[c])
        summed[c]=dict(summary)
    eligible=[c for c,s in summed.items() if s["correct"]>=.70*s["accepted"]]
    best=max(eligible,key=lambda c:(summed[c]["correct"],-(summed[c]["accepted"]-summed[c]["correct"]),c))
    return best,fits,summed


def run():
    observer,settings,sources=load_observer()
    control=PREV/"selector.joblib"
    assert old.digest(control)==old.read(PREV/"training.json")["checkpoint_sha256"]
    sources.update((Path(__file__),Path(features.__code__.co_filename),Path(read_development.__code__.co_filename),
                    Path(patch.__code__.co_filename),Path(selected.__code__.co_filename),control,PREV/"training.json"))
    dev={}
    for track in DEVELOP:
        sensor,truth,receipt=read_development(track,sources,settings)
        edge,visible=load_labels(EXTRA/f"{track}-train-labels.npz",sensor,sources,receipt["camera_bundle_sha256_by_run"])
        dev[track]=dict(sensors=sensor,training=(bank(SensorArrays(sensor),settings),edge,visible))
    cut,folds,summed=development(dev)
    model=fit([v["training"] for v in dev.values()])
    model_file=OUT/"selector.joblib"
    joblib.dump(model,model_file)
    assert all(t not in str(p) for p in sources for t,_ in EVALUATION)
    calibration=OUT/"training.json"
    old.write(calibration,dict(development=list(dev),excluded=[t for t,_ in EVALUATION],threshold=cut,
              precision_floor=.70,folds=folds,totals=summed,checkpoint_sha256=old.digest(model_file),seed=SEED,
              source_revision=old.revision(),source_sha256={str(p):old.digest(p) for p in sorted(sources)},
              termination="frozen_before_replay"))
    print("Frozen",cut,summed,flush=True)
    previous=joblib.load(control)["model"]
    reports={}
    for track,folder in EVALUATION:
        root=ARCHIVE/folder
        source=root/"sensors.npz"
        receipt=root/"result.json"
        assert old.digest(source)==old.read(receipt)["inputs_sha256"]
        with np.load(source,allow_pickle=False) as d:
            sensor=SensorArrays({k:d[k] for k in d.files})
        raw=sensor["sections"]
        b=bank(sensor,settings)
        chosen=choose(b,model)
        changed=apply(raw,b,chosen,cut)
        ex=examples(features(sensor,settings)[0])
        prior=patch(raw,ex,selected(ex,previous),POLICIES["score_65"])
        archive=root/"predictions/forecasts.jsonl"
        base=frames(archive,old.read(archive.with_name("result.json"))["forecasts_sha256"])
        truth,path=reference(track)
        proofs={source,receipt,archive,path}
        modes={}
        for key,edges in (("prior",prior),("width",changed)):
            revised=observe(SensorArrays(dict(sensor,sections=edges)),observer,settings)[0]
            # Whole-observer comparison includes candidate changes and can change original alerts.
            output=OUT/f"{track}-{key}-frames.jsonl"
            labels=row_map(truth)
            scored=[r for r in revised if (r["run"],r["frame"]) in labels]
            assert len(scored)==len(truth)
            metrics=forecast_errors(scored,truth)
            records=[dict(run=r["run"],frame=r["frame"],candidate=r["candidate"],forecasts=r["forecasts"]) for r in scored]
            old.jsonl(output,records)
            modes[key]=dict(metrics=metrics,trace_sha256=old.digest(output),points=int((~np.isfinite(raw)&np.isfinite(edges)).sum()))
        report=dict(modes=modes)
        coordinates=OUT/f"{track}-boundaries.npz"
        np.savez_compressed(coordinates,run=sensor["run"],frame=sensor["frame"],stations=sensor["stations"],
                            observed=raw,prior=prior,width=changed)
        report["boundaries_sha256"]=old.digest(coordinates)
        if track==SILVER:
            label_file=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            target,seen=load_labels(label_file,sensor,proofs)
            labels=row_map(truth)
            right=np.array([labels[r["run"],r["frame"]]["truth"]=="approach" and
                            labels[r["run"],r["frame"]]["label"]=="2" for r in base])
            assert right.sum()==13
            report["right_approach"]={key:{side:boundary_scores(raw[right,:,j],edges[right,:,j],
                target[right,:,j],seen[right,:,j]) for j,side in enumerate(("right","left"))}
                for key,edges in (("prior",prior),("width",changed))}
            i,j,k=b["positions"].T
            focused=right[i]&(k==0)&seen[i,j,k]
            prob,val,origin=chosen
            report["missing_right_points"]=[dict(frame=int(sensor["frame"][i[x]]),station_m=float(sensor["stations"][j[x]]),
                source=int(origin[x]),score=float(prob[x]),error_m=float(abs(val[x]*SIGNS[k[x]]-target[i[x],j[x],k[x]])),
                adopted=bool(np.isfinite(changed[i[x],j[x],k[x]]))) for x in np.flatnonzero(focused)]
        report["source_sha256"]={str(p):old.digest(p) for p in sorted(proofs)}
        reports[track]=report
        print(track,report.get("right_approach"),modes,flush=True)
    old.write(OUT/"result.json",dict(tracks=reports,training_sha256=old.digest(calibration),termination="paired_scored"))


if __name__=="__main__":
    run()
