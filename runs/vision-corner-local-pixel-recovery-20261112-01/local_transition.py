"""Search for current-image edge transitions near a geometry-based width proposal."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
PIXEL=OUT.parent/"vision-corner-pixel-checked-boundary-20261112-01"
sys.path.insert(0,str(PIXEL))
from pixel_checked import (ARCHIVE, DEVELOP, EVALUATION, EXTRA, METHODS, N, SEED, SIGNS, SILVER, WIDTH, SensorArrays,
    apply, bank, boundary_scores, choose, forecast_errors, frames, joblib, load_labels, load_observer,
    observe, old, pixel_bank, read_development, reference, row_map)
from width_conditioned import fit

SHIFTS=np.array([-.75,-.5,-.25,0.,.25,.5,.75])
MODES=("contrast","localized")
CUTS=(10.,15.,20.,25.,30.,35.,40.,50.,60.,70.,80.,100.,400.)


def search(track,sensor,b,proof,inputs):
    width_rows=np.flatnonzero(b["source"]==5)
    original_cells=b["cell"][width_rows]
    n=len(SHIFTS)
    row=np.repeat(width_rows,n)
    step=np.tile(SHIFTS,len(width_rows))
    lattice=dict(b,cell=b["cell"][row],source=b["source"][row],value=b["value"][row]+step,
                 x=b["x"][row],counts=b["counts"][row])
    _,channels=pixel_bank(track,sensor,lattice,proof,inputs)
    profile=channels.reshape(len(row),2,N)
    brightness=np.maximum(-profile[:,:,0],0.)
    center=profile[:,:,6]
    maximum=profile[:,:,7]
    # Each channel must itself cover the entire strip.
    strength=np.where(np.isfinite(brightness),brightness,-np.inf)
    joint=np.sqrt(brightness*center)*np.minimum(1.,(center+1.)/(maximum+1.))
    local=np.where(np.isfinite(joint),joint,-np.inf)
    # Prefer a nearby transition when nearby image measurements are similar.
    scores={"contrast":strength.max(axis=1)-12.*abs(step),"localized":local.max(axis=1)-12.*abs(step)}
    result={}
    for mode,p in scores.items():
        matrix=p.reshape(-1,n)
        best=np.argmax(matrix,axis=1)
        flat=np.arange(len(best))*n+best
        assert np.array_equal(lattice["cell"][flat],original_cells)
        view=np.argmax((strength if mode=="contrast" else local)[flat],axis=1)
        result[mode]=dict(pos=b["positions"][original_cells],coordinate=lattice["value"][flat],
                          score=p[flat],shift=step[flat],view=view,channels=profile[flat])
    return result


def add(raw,prior,options,cut):
    result=prior.copy()
    i,j,k=options["pos"].T
    chosen=(options["score"]>=cut)&~np.isfinite(prior[i,j,k])
    result[i[chosen],j[chosen],k[chosen]]=options["coordinate"][chosen]*SIGNS[k[chosen]]
    width=result[:,:,1]-result[:,:,0]
    invalid=np.isfinite(width)&((width<=3.)|(width>=20.))
    result[invalid]=prior[invalid]
    np.testing.assert_array_equal(result[np.isfinite(prior)],prior[np.isfinite(prior)])
    return result


def measure(prior,updated,labels,valid):
    use=~np.isfinite(prior)&np.isfinite(updated)&valid
    errors=abs(updated[use]-labels[use])
    return dict(accepted=int(use.sum()),correct=int((errors<=1.25).sum()),error_m=float(errors.sum()))


def develop(data,prior_cut):
    totals={m:{c:Counter() for c in CUTS} for m in MODES}
    folds={}
    for held,item in data.items():
        raw=item["raw"]
        bank,labels,valid=item["training"]
        control=fit([o["training"] for t,o in data.items() if t!=held])
        existing=apply(raw,bank,choose(bank,control),prior_cut)
        folds[held]={}
        for mode,options in item["options"].items():
            report={}
            for c in CUTS:
                summary=measure(existing,add(raw,existing,options,c),labels,valid)
                totals[mode][c].update(summary)
                report[c]=summary
            folds[held][mode]=report
        print(held,{mode:{c:p for c,p in by.items() if c in (20.,40.,60.)} for mode,by in folds[held].items()},flush=True)
    def priority(entry):
        mode,c=entry
        s=totals[mode][c]
        bad=s["accepted"]-s["correct"]
        return s["correct"]-2*bad,s["correct"],c
    best=max(((m,c) for m in MODES for c in CUTS),key=priority)
    return best,folds,totals


def run():
    observer,settings,inputs=load_observer()
    frozen=old.read(WIDTH/"training.json")
    checkpoint=WIDTH/"selector.joblib"
    assert old.digest(checkpoint)==frozen["checkpoint_sha256"]
    inputs.update((Path(__file__),Path(bank.__code__.co_filename),Path(pixel_bank.__code__.co_filename),
                   Path(read_development.__code__.co_filename),WIDTH/"training.json",checkpoint))
    data={}
    for track in DEVELOP:
        sensors,_,proof=read_development(track,inputs,settings)
        sensors=SensorArrays(sensors)
        original=bank(sensors,settings)
        labels,valid=load_labels(EXTRA/f"{track}-train-labels.npz",sensors,inputs,proof["camera_bundle_sha256_by_run"])
        choices=search(track,sensors,original,proof["camera_bundle_sha256_by_run"],inputs)
        data[track]=dict(raw=sensors["sections"],training=(original,labels,valid),options=choices)
    (mode,cut),folds,totals=develop(data,frozen["threshold"])
    assert all(t not in str(p) for p in inputs for t,_ in EVALUATION)
    training=OUT/"training.json"
    old.write(training,dict(development=list(data),excluded=[t for t,_ in EVALUATION],mode=mode,threshold=cut,
              folds=folds,totals=totals,source_revision=old.revision(),seed=SEED,
              source_sha256={str(p):old.digest(p) for p in sorted(inputs)},termination="frozen_before_replay"))
    print("Frozen",mode,cut,totals[mode][cut],flush=True)
    control=joblib.load(checkpoint)
    reports={}
    for track,folder in EVALUATION:
        root=ARCHIVE/folder
        receipt=root/"result.json"
        proof=old.read(receipt)
        source=root/"sensors.npz"
        assert old.digest(source)==proof["inputs_sha256"]
        with np.load(source,allow_pickle=False) as f:
            sensor=SensorArrays({k:f[k] for k in f.files})
        raw=sensor["sections"]
        paths={source,receipt}
        original=bank(sensor,settings)
        candidates=search(track,sensor,original,proof["archive_bundles"],paths)
        base=apply(raw,original,choose(original,control),frozen["threshold"])
        repaired=add(raw,base,candidates[mode],cut)
        truth,path=reference(track)
        paths.add(path)
        labels=row_map(truth)
        metrics={}
        for key,edges in (("base",base),("local",repaired)):
            replay=observe(SensorArrays(dict(sensor,sections=edges)),observer,settings)[0]
            rows=[r for r in replay if (r["run"],r["frame"]) in labels]
            output=OUT/f"{track}-{key}-frames.jsonl"
            old.jsonl(output,rows)
            metrics[key]=dict(counts=forecast_errors(rows,truth),trace_sha256=old.digest(output))
        bundle=OUT/f"{track}-boundaries.npz"
        np.savez_compressed(bundle,run=sensor["run"],frame=sensor["frame"],stations=sensor["stations"],
                            raw=raw,base=base,local=repaired)
        info=dict(metrics=metrics,boundary_sha256=old.digest(bundle),
                  extra_points=int((~np.isfinite(base)&np.isfinite(repaired)).sum()))
        if track==SILVER:
            mesh=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge,visible=load_labels(mesh,sensor,paths)
            right=np.array([labels[r,f]["truth"]=="approach" and labels[r,f]["label"]=="2"
                            for r,f in zip(sensor["run"],sensor["frame"],strict=True)])
            assert right.sum()==13
            info["right_approach"]={name:{side:boundary_scores(raw[right,:,j],edges[right,:,j],
                    edge[right,:,j],visible[right,:,j]) for j,side in enumerate(("right","left"))}
                    for name,edges in (("base",base),("local",repaired))}
            alternatives={}
            for m,o in candidates.items():
                i,j,k=o["pos"].T
                eligible=right[i]&(k==0)&visible[i,j,k]
                alternatives[m]=[dict(frame=int(sensor["frame"][i[t]]),station_m=float(sensor["stations"][j[t]]),
                    score=float(o["score"][t]) if np.isfinite(o["score"][t]) else None,
                    error_m=float(abs(o["coordinate"][t]*SIGNS[k[t]]-edge[i[t],j[t],k[t]])),
                    shift_m=float(o["shift"][t]),view=("front","outward")[int(o["view"][t])])
                    for t in np.flatnonzero(eligible)]
            info["local_diagnostics"]=alternatives
        info["source_sha256"]={str(p):old.digest(p) for p in sorted(paths)}
        reports[track]=info
        print(track,info["extra_points"],info.get("right_approach"),metrics,flush=True)
    old.write(OUT/"result.json",dict(tracks=reports,training_sha256=old.digest(training),termination="paired_scored"))


if __name__=="__main__":
    run()
