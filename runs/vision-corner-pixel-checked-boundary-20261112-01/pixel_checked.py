"""Score missing road-edge hypotheses against the current calibrated camera frame."""
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
WIDTH=OUT.parent/"vision-corner-width-conditioned-boundary-20261112-01"
sys.path.insert(0,str(WIDTH))
from width_conditioned import (ARCHIVE, DEVELOP, EVALUATION, EXTRA, METHODS, SEED, SIGNS, SILVER, SensorArrays,
    apply, bank, boundary_scores, choose, development, fit, forecast_errors, frames, joblib, load_labels,
    load_observer, observe, old, read_development, reference, row_map)
from replay_corner_camera_archive import stream, image

RUNS=OUT.parent
MAP={
    "imola":RUNS/"vision-imola-approach-capture-20261111-02/archive-map.json",
    "ks_laguna_seca":RUNS/"vision-laguna-independent-capture-20261111-01/archive-map.json",
    "ks_zandvoort":RUNS/"vision-zandvoort-independent-capture-20261111-02/archive-map.json",
    "mugello":RUNS/"vision-mugello-independent-capture-20261111-01/archive-map.json",
    SILVER:RUNS/"vision-silverstone1967-independent-capture-20261112-05/archive-map.json",
    "ks_vallelunga":RUNS/"vision-vallelunga-local-corner-capture-20261112-01/archive-map.json",
    "monza":ARCHIVE/"monza-transfer-inputs/archive-map.json",
}
OFFSET=np.array([-.8,-.5,-.25,0.,.25,.5,.8])
VIEWS=("front","outward")
N=9


def strip_features(pixels,camera,points):
    delta=points-np.asarray(camera["origin_body_m"])
    depth=delta @ camera["forward_body"]
    safe=np.where(depth>0,depth,1.)
    u=.5+.5*camera["projection_x"]*(delta @ camera["right_body"])/safe
    v=.5-.5*camera["projection_y"]*(delta @ camera["up_body"])/safe
    valid=(depth>0)&(u>=.02)&(u<=.98)&(v>=.02)&(v<=.98)
    h,w=pixels.shape[:2]
    cols=np.rint(np.clip(u,0,1)*(w-1)).astype(int)
    rows=np.rint(np.clip(v,0,1)*(h-1)).astype(int)
    spans=np.hypot(cols[:,-1]-cols[:,0],rows[:,-1]-rows[:,0])
    keep=valid.all(axis=1)&(spans>=2.)
    values=np.full((len(points),N),np.nan)
    if not keep.any():
        return values
    samples=pixels[rows[keep],cols[keep]].astype(float)
    inside=np.median(samples[:,:3],axis=1)
    outside=np.median(samples[:,4:],axis=1)
    d=outside-inside
    local=np.diff(samples,axis=1)
    strength=np.linalg.norm(local,axis=-1)
    chroma=np.linalg.norm(d[:,1:],axis=-1)
    inner=np.linalg.norm(inside[:,1:]-128.,axis=-1)
    outer=np.linalg.norm(outside[:,1:]-128.,axis=-1)
    values[keep]=np.column_stack((d[:,0],abs(d[:,0]),chroma,inner,outer,outer-inner,
                                  strength[:,2:4].max(1),strength.max(1),spans[keep]))
    return values


def pixel_bank(track,sensor,base,bundles,sources):
    mapping=MAP[track]
    folders=old.read(mapping)
    assert set(folders)==set(bundles)
    mount=old.read(old.MOUNT)["views"]
    lookup={(str(run),int(frame)):i for i,(run,frame) in enumerate(zip(sensor["run"],sensor["frame"],strict=True))}
    assert len(lookup)==len(sensor["frame"])
    physical=base["positions"][base["cell"]]
    i,j,k=physical.T
    order=np.argsort(i,kind="stable")
    bounds=np.searchsorted(i[order],np.arange(len(lookup)+1))
    channels=np.full((len(i),len(VIEWS)*N),np.nan)
    seen=set()
    for run,folder in folders.items():
        for archive,item,meta,views in stream(folder,bundles[run],track):
            index=lookup[run,item["capture_id"]]
            assert index not in seen
            seen.add(index)
            active=order[bounds[index]:bounds[index+1]]
            if views is None or not len(active):
                continue
            values=base["value"][active,None]+OFFSET[None,:]
            y=values*SIGNS[k[active],None]
            x=np.broadcast_to(sensor["stations"][j[active],None],y.shape)
            ground=np.stack((x,y,np.zeros_like(x)),axis=-1)
            cache={}
            for side,name in enumerate(("right","left")):
                part=k[active]==side
                if not part.any():
                    continue
                for m,view in enumerate(("front",name)):
                    key=active[part]
                    # Cache per frame because the front view can cover both sides.
                    lab=cache.get(view)
                    if lab is None:
                        lab=cv2.cvtColor(image(archive,views[view]),cv2.COLOR_BGR2LAB)
                        cache[view]=lab
                    channels[key,m*N:(m+1)*N]=strip_features(lab,mount[view],ground[part])
    assert len(seen)==len(lookup)
    sources.update((mapping,old.MOUNT,Path(stream.__code__.co_filename),Path(image.__code__.co_filename)))
    observed=int(np.isfinite(channels[:,0::N]).any(axis=1).sum())
    print("Image features",track,observed,len(i),flush=True)
    return dict(base,x=np.column_stack((base["x"],channels))),channels


def run():
    observer,settings,sources=load_observer()
    sources.update((Path(__file__),Path(bank.__code__.co_filename),Path(fit.__code__.co_filename),
                    Path(read_development.__code__.co_filename),WIDTH/"training.json"))
    cv2.setNumThreads(1)
    dev={}
    for track in DEVELOP:
        sensor,_,proof=read_development(track,sources,settings)
        sensor=SensorArrays(sensor)
        labels,valid=load_labels(EXTRA/f"{track}-train-labels.npz",sensor,sources,proof["camera_bundle_sha256_by_run"])
        base=bank(sensor,settings)
        data,_=pixel_bank(track,sensor,base,proof["camera_bundle_sha256_by_run"],sources)
        dev[track]=dict(sensors=sensor,training=(data,labels,valid))
    cut,folds,summed=development(dev)
    model=fit([v["training"] for v in dev.values()])
    previous=old.read(WIDTH/"training.json")
    control_file=WIDTH/"selector.joblib"
    assert old.digest(control_file)==previous["checkpoint_sha256"]
    sources.add(control_file)
    assert all(t not in str(p) for p in sources for t,_ in EVALUATION)
    output=OUT/"selector.joblib"
    joblib.dump(model,output)
    frozen=OUT/"training.json"
    old.write(frozen,dict(development=list(dev),excluded=[t for t,_ in EVALUATION],threshold=cut,
              control_threshold=previous["threshold"],folds=folds,totals=summed,precision_floor=.70,
              seed=SEED,checkpoint_sha256=old.digest(output),source_revision=old.revision(),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="frozen_before_replay"))
    print("Frozen",cut,summed[cut],flush=True)
    control=joblib.load(control_file)
    results={}
    for track,folder in EVALUATION:
        root=ARCHIVE/folder
        receipt=root/"result.json"
        proof=old.read(receipt)
        path=root/"sensors.npz"
        assert old.digest(path)==proof["inputs_sha256"]
        with np.load(path,allow_pickle=False) as d:
            sensor=SensorArrays({k:d[k] for k in d.files})
        raw=sensor["sections"]
        evidence={path,receipt}
        base=bank(sensor,settings)
        data,channels=pixel_bank(track,sensor,base,proof["archive_bundles"],evidence)
        options=dict(width=apply(raw,base,choose(base,control),previous["threshold"]))
        picked=choose(data,model)
        options["pixels"]=apply(raw,data,picked,cut)
        labels,label_path=reference(track)
        keys=row_map(labels)
        evidence.add(label_path)
        outputs={}
        for mode,edges in options.items():
            forecasts=observe(SensorArrays(dict(sensor,sections=edges)),observer,settings)[0]
            rows=[r for r in forecasts if (r["run"],r["frame"]) in keys]
            metrics=forecast_errors(rows,labels)
            trace=OUT/f"{track}-{mode}-frames.jsonl"
            old.jsonl(trace,rows)
            outputs[mode]=dict(metrics=metrics,trace_sha256=old.digest(trace),
                              points=int((~np.isfinite(raw)&np.isfinite(edges)).sum()))
        report=dict(modes=outputs)
        boundary=OUT/f"{track}-boundaries.npz"
        np.savez_compressed(boundary,run=sensor["run"],frame=sensor["frame"],stations=sensor["stations"],
                            observed=raw,**options)
        report["boundaries_sha256"]=old.digest(boundary)
        if track==SILVER:
            mesh=RUNS/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge,visible=load_labels(mesh,sensor,evidence)
            target=np.array([keys[run,frame]["truth"]=="approach" and keys[run,frame]["label"]=="2"
                             for run,frame in zip(sensor["run"],sensor["frame"],strict=True)])
            assert target.sum()==13
            report["right_approach"]={mode:{side:boundary_scores(raw[target,:,j],edges[target,:,j],edge[target,:,j],
                        visible[target,:,j]) for j,side in enumerate(("right","left"))} for mode,edges in options.items()}
            i,j,k=data["positions"].T
            interest=target[i]&(k==0)&visible[i,j,k]
            visits=[]
            for c in np.flatnonzero(interest):
                ix=np.flatnonzero(data["cell"]==c)
                source=picked[2][c]
                row=ix[np.flatnonzero(data["source"][ix]==source)[0]]
                visits.append(dict(frame=int(sensor["frame"][i[c]]),station_m=float(sensor["stations"][j[c]]),
                    source=int(source),score=float(picked[0][c]),error_m=float(abs(picked[1][c]*SIGNS[k[c]]-edge[i[c],j[c],k[c]])),
                    adopted=bool(np.isfinite(options["pixels"][i[c],j[c],k[c]])),
                    views={view:channels[row,v*N:(v+1)*N].tolist() if np.isfinite(channels[row,v*N]) else None
                           for v,view in enumerate(VIEWS)}))
            report["right_missing_points"]=visits
        report["source_sha256"]={str(p):old.digest(p) for p in sorted(evidence)}
        results[track]=report
        print(track,report.get("right_approach"),outputs,flush=True)
    old.write(OUT/"result.json",dict(tracks=results,training_sha256=old.digest(frozen),termination="paired_scored"))


if __name__=="__main__":
    run()
