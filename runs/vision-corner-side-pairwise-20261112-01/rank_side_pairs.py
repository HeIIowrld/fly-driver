"""Rank neighbouring side-image transitions against the physical road edge."""
from pathlib import Path
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent/"vision-corner-side-transition-20261112-01"
sys.path.insert(0,str(PREVIOUS))
from rank_transitions import SIDE, CACHE, DEV, TEST, VAL, SEED, STATIONS, DISTANCES, SIDES, TOLERANCE_M
from rank_transitions import (old, image, stream, sampling, image_features, labels, baseline_data,
    RAW_ARCHIVE, ROUTING, CAPTURE, MESH, metrics, mesh_score, guard, across_normal, load_observer, observe, frames)
from listwise import objective
from replay_corner_boundary_states import joblib
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

CLOSE_M = .6
FEATURE_MODES = ("geometry_and_contrast", "with_signed_strips")


def candidate_features(index, s, score, separation, grid, photometry):
    splits = np.split(index,np.flatnonzero(np.diff(index)>1)+1)
    support = np.zeros((len(grid),2))
    for segment in splits:
        support[segment,0] = len(segment)*.2
        support[segment,1] = (np.arange(len(segment))-(len(segment)-1)/2)*.2
    best = index[np.argmax(score[index])]
    dist = abs(grid[index])
    geometry = np.c_[score[index],score[index]-score[best],dist,dist-abs(grid[best]),
                    separation[index],np.full(len(index),STATIONS[s]),support[index]]
    return np.c_[geometry,photometry[index]]


def candidates(track,mapping,bundles,keys,observed,motion=None):
    lookup = sampling(old.read(old.MOUNT))
    index = {key:i for i,key in enumerate(keys)}
    seen = set()
    for name,folder in old.read(mapping).items():
        for archive,item,_,views in stream(folder,bundles[name],track):
            key = name,item["capture_id"]
            if key not in index:
                continue
            i = index[key]
            assert i not in seen
            seen.add(i)
            if not views or (motion is not None and (not np.isfinite(motion[i]).all() or motion[i,1]<old.MIN_SPEED_MPS)):
                continue
            for side,view in enumerate(SIDES):
                missing = np.flatnonzero(~np.isfinite(observed[i,:len(STATIONS),side]))
                if not len(missing):
                    continue
                bitmap = image(archive,views[view])
                signed,valid = image_features(bitmap,lookup[view])
                u,v,mask = lookup[view]
                normal = across_normal(cv2.cvtColor(bitmap,cv2.COLOR_BGR2LAB),u/319.,v/179.,mask)
                grid = (-1.,1.)[side]*DISTANCES
                for s in missing:
                    possible = valid[s] & guard(observed[i],old.STATIONS,s,side,grid,True)
                    finite = np.isfinite(normal[s])
                    possible &= finite & (np.r_[False,finite[:-1]] | np.r_[finite[1:],False])
                    keep = np.flatnonzero(possible)
                    if not len(keep):
                        continue
                    near = np.isfinite(observed[i,:,side]) & (abs(old.STATIONS-old.STATIONS[s])<=4.)
                    distance = np.min(abs(grid[:,None]-observed[i,near,side]),axis=1)
                    x = candidate_features(keep,s,normal[s],distance,grid,signed[s])
                    assert np.isfinite(x).all()
                    yield (i,s,side),grid[keep],x
    assert len(seen)==len(keys)


def development(track,hashes):
    group,_,mapping,bundles=baseline_data(track,hashes)
    keys=[(r["run"],r["frame"]) for r in group["rows"]]
    path=CACHE/f"{track}-sections.npz"
    proof=path.with_suffix(".json")
    assert old.digest(path)==old.read(proof)["sections_sha256"]
    with np.load(path,allow_pickle=False) as d:
        assert keys==list(zip(d["run"],d["frame"],strict=True))
        assert np.array_equal(d["stations"],old.STATIONS[:len(d["stations"])])
        current=d["values"][:,0]
    sections=np.full((len(keys),len(old.STATIONS),2),np.nan)
    sections[:,:current.shape[1]]=current
    lab,label_path=labels(track)
    assert keys==list(zip(lab["run"],lab["frame"],strict=True))
    cam=np.stack((lab["image_px"][:,2,:,0],lab["image_px"][:,1,:,1]),axis=2)
    visible=lab["adjacent_invalid"] & np.isfinite(cam).all(-1)
    output=[]
    for index,grid,x in candidates(track,mapping,bundles,keys,sections):
        if visible[index]:
            output.append((grid,x,float(lab["lateral_m"][index])))
    hashes.update((path,proof,label_path,label_path.with_suffix(".json"),mapping))
    return output


def feature_view(x,mode):
    return x if mode=="with_signed_strips" else x[:,:8]


def fit(groups,mode):
    by_track={}
    for track,rows in groups.items():
        by_track[track]=[(feature_view(x,mode),abs(grid-target)<=CLOSE_M)
                        for grid,x,target in rows if (abs(grid-target)<=CLOSE_M).any()
                        and (abs(grid-target)>CLOSE_M).any()]
    samples=[(x,good,1./(len(by_track)*len(rows))) for rows in by_track.values() for x,good in rows]
    raw=np.concatenate([x for x,_,_ in samples])
    weights=np.concatenate([np.full(len(x),w/len(x)) for x,_,w in samples])
    scale=StandardScaler().fit(raw,sample_weight=weights)
    samples=[(np.clip(scale.transform(x),-6.,6.),good,w) for x,good,w in samples]
    result=minimize(objective,np.zeros(raw.shape[1]),args=(samples,),method="L-BFGS-B",jac=True,
                    options=dict(maxiter=100,gtol=1.e-6,ftol=1.e-10))
    assert result.success,result.message
    return dict(weights=result.x,scale=scale,mode=mode,groups={t:len(rows) for t,rows in by_track.items()},
                loss=float(result.fun),iterations=int(result.nit))


def select_candidate(grid,x,model):
    scores=x[:,0] if model is None else np.clip(model["scale"].transform(feature_view(x,model["mode"])),
                                                -6.,6.) @ model["weights"]
    return grid[np.argmax(scores)]


def stats(rows,model):
    signed=np.array([abs(select_candidate(grid,x,model))-abs(y) for grid,x,y in rows])
    possible=sum(np.min(abs(grid-y))<=TOLERANCE_M for grid,x,y in rows)
    return dict(points=len(rows),available_within_1_25_m=int(possible), mae_m=float(abs(signed).mean()),
                signed_m=float(signed.mean()),within_1_25_m=int((abs(signed)<=TOLERANCE_M).sum()))


def patch(raw,entries,model):
    sections=raw.copy()
    for index,grid,x in entries:
        assert not np.isfinite(raw[index])
        sections[index]=select_candidate(grid,x,model)
    width=sections[:,:,1]-sections[:,:,0]
    invalid=np.isfinite(width)&((width<=3.)|(width>=20.))
    sections[invalid]=raw[invalid]
    np.testing.assert_array_equal(sections[np.isfinite(raw)],raw[np.isfinite(raw)])
    return sections


def main():
    cv2.setNumThreads(1)
    sources={Path(__file__),old.MOUNT,Path(sampling.__code__.co_filename),Path(objective.__code__.co_filename),
             Path(across_normal.__code__.co_filename),Path(guard.__code__.co_filename),Path(stream.__code__.co_filename)}
    groups={track:development(track,sources) for track in DEV}
    baseline={t:stats(rows,None) for t,rows in groups.items()}
    evaluated={}
    for mode in FEATURE_MODES:
        evaluated[mode]={held:stats(rows,fit({k:v for k,v in groups.items() if k!=held},mode))
                         for held,rows in groups.items()}
    scores={"baseline":baseline,**evaluated}
    winner=min(scores,key=lambda m:(max(scores[m][t]["mae_m"]/baseline[t]["mae_m"] for t in DEV),
                                    sum(scores[m][t]["mae_m"]/baseline[t]["mae_m"] for t in DEV)))
    fitted={mode:fit(groups,mode) for mode in FEATURE_MODES}
    checkpoint=OUT/"rankers.joblib"
    joblib.dump(fitted,checkpoint)
    training=OUT/"training.json"
    assert not any(TEST in str(p) or VAL in str(p) for p in sources)
    old.write(training,dict(status="frozen_before_replay",development=DEV,excluded=[TEST,VAL],seed=SEED,
                source_revision=old.revision(),close_training_m=CLOSE_M,winner=winner,fold_scores=scores,
                model_groups={m:v["groups"] for m,v in fitted.items()},checkpoint_sha256=old.digest(checkpoint),
                source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="paired_candidate_training"))
    print("Frozen",winner,scores,flush=True)
    models,settings,pins=load_observer()
    prior=old.read(ROUTING/"straight-verification.json")
    reports={}
    for track,name,mapping in (
            (TEST,"silverstone",CAPTURE),
            (VAL,"vallelunga",OUT.parent/"vision-vallelunga-local-corner-capture-20261112-01"/"archive-map.json")):
        directory=RAW_ARCHIVE/name
        original=directory/"sensors.npz"
        raw_proof=directory/"result.json"
        receipt=old.read(raw_proof)
        assert old.digest(original)==receipt["inputs_sha256"]
        with np.load(original,allow_pickle=False) as arrays:
            raw={key:arrays[key] for key in arrays.files}
        keys=list(zip(raw["run"],raw["frame"],strict=True))
        options=list(candidates(track,mapping,receipt["archive_bundles"],keys,raw["sections"],raw["motion"]))
        control=patch(raw["sections"],options,None)
        control_path=SIDE/f"{track}-normal_anchor-sensors.npz"
        old_side=old.read(SIDE/"side-result.json")
        assert old.digest(control_path)==old_side["tracks"][track]["methods"]["normal_anchor"]["sensor_sha256"]
        with np.load(control_path,allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sections"],control)
        reference_path=ROUTING/f"{track}-state-distance.jsonl"
        reference=frames(reference_path,prior["tracks"][track]["rows_sha256"])
        report={}
        for mode,fit_model in (("baseline",None),*fitted.items()):
            selected=patch(raw["sections"],options,fit_model)
            sensors=OUT/f"{track}-{mode}-sensors.npz"
            np.savez_compressed(sensors,**dict(raw,sections=selected))
            with np.load(sensors,allow_pickle=False) as arrays:
                rows,aligned,_=observe(arrays,models,settings)
            trace=OUT/f"{track}-{mode}-frames.jsonl"
            old.jsonl(trace,rows)
            metrics_by_method,right=metrics(rows,reference)
            result=dict(metrics=metrics_by_method,right_hits=right,frames_sha256=old.digest(trace),
                        sensors_sha256=old.digest(sensors))
            if mode=="baseline":
                assert result["frames_sha256"]==old_side["tracks"][track]["methods"]["normal_anchor"]["frames_sha256"]
            if track==TEST:
                with np.load(directory/"predictions/boundaries.npz",allow_pickle=False) as base:
                    result["mesh"]=mesh_score(aligned,base["estimated"],reference,keys)
            report[mode]=result
        sources={*pins,training,checkpoint,original,raw_proof,mapping,control_path,reference_path}
        if track==TEST:
            sources.update((MESH,MESH.with_suffix(".json"),directory/"predictions/boundaries.npz"))
        reports[track]=dict(methods=report,source_sha256={str(p):old.digest(p) for p in sorted(sources)})
        print(track,{m:(v["right_hits"],v.get("mesh",{}).get("new_mae_m"))
                     for m,v in report.items()},flush=True)
    old.write(OUT/"result.json",dict(status="paired_models_replayed",training_sha256=old.digest(training),
                  tracks=reports,termination="side_rankers_scored_on_same_videos"))


if __name__=="__main__":
    main()
