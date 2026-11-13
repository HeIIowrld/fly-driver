"""Select a continuous side boundary from the existing image-supported candidates."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import sys

import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
PAIRS=OUT.parent/"vision-corner-side-pairwise-20261112-01"
sys.path.insert(0,str(PAIRS))
from rank_side_pairs import (SIDE,CACHE,DEV,TEST,VAL,SEED,old,baseline_data,labels,candidates,
    RAW_ARCHIVE,ROUTING,CAPTURE,MESH,metrics,mesh_score,load_observer,observe,frames)

TRIALS=tuple(product((0.,.25,1.,3.),(0.,.25,1.)))
BASELINE=(0.,0.)
CONTRAST_SCALE=10.


def huber(v):
    a=abs(v)
    return np.where(a<=.5,.5*a*a,.5*(a-.25))


def solve(stations,grid,unary,prior,penalty):
    """Maximise visual scores plus first- and second-order continuity."""
    slope_weight,curve_weight=penalty
    if len(grid)==1 or penalty==BASELINE:
        return [int(np.argmax(v)) for v in unary]
    span=np.diff(stations)
    slopes=[(b[None,:]-a[:,None])/d for a,b,d in zip(grid,grid[1:],span,strict=False)]
    pair=unary[0][:,None]+unary[1][None,:]-slope_weight*huber(slopes[0]-prior)
    back=[]
    for s in range(1,len(slopes)):
        delta=slopes[s][None,:,:]-slopes[s-1][:,:,None]
        potential=pair[:,:,None]-curve_weight*huber(delta)
        best=potential.argmax(axis=0)
        pair=np.take_along_axis(potential,best[None,:,:],axis=0)[0]
        pair+=unary[s+1][None,:]-slope_weight*huber(slopes[s]-prior)
        back.append(best)
    a,b=np.unravel_index(pair.argmax(),pair.shape)
    path=[int(b),int(a)]
    for previous in reversed(back):
        path.append(int(previous[path[-1],path[-2]]))
    return path[::-1]


def groups(observed,entries):
    choices=defaultdict(dict)
    for (frame,station,side),grid,features in entries:
        choices[frame,side][station]=(grid,features[:,0])
    for (frame,side),row in choices.items():
        original=observed[frame,:,side]
        known=np.flatnonzero(np.isfinite(original))
        positions=np.array(sorted(row))
        chains=np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)
        for chain in chains:
            start,end=chain[0],chain[-1]
            before=known[(known<start)&(start-known<=2)]
            after=known[(known>end)&(known-end<=2)]
            anchors=np.r_[before[-1:],after[:1]]
            window=np.r_[before[-1:],chain,after[:1]]
            if len(anchors)>=2:
                distance=old.STATIONS[anchors[-1]]-old.STATIONS[anchors[0]]
                slope=float((original[anchors[-1]]-original[anchors[0]])/distance)
            else:
                closest=known[abs(known-.5*(start+end))<=4]
                if len(closest)>=2:
                    slopes=np.diff(original[closest])/np.diff(old.STATIONS[closest])
                    slope=float(np.median(slopes))
                else:
                    slope=0.
            x=[]; confidence=[]
            for j in window:
                if j in row:
                    grid,score=row[j]
                    x.append(grid)
                    confidence.append((score-score.max())/CONTRAST_SCALE)
                else:
                    x.append(np.array([original[j]]))
                    confidence.append(np.zeros(1))
            yield (frame,side,window,x,confidence,float(np.clip(slope,-1.,1.)),set(chain.tolist()))


def patch(raw,grouped,policy):
    selected=raw.copy()
    for frame,side,window,grids,scores,slope,chain in grouped:
        choice=solve(old.STATIONS[window],grids,scores,slope,policy)
        for station,grid,j in zip(window,grids,choice,strict=True):
            if station in chain:
                selected[frame,station,side]=grid[j]
    width=selected[:,:,1]-selected[:,:,0]
    invalid=np.isfinite(width)&((width<=3.)|(width>=20.))
    selected[invalid]=raw[invalid]
    np.testing.assert_array_equal(selected[np.isfinite(raw)],raw[np.isfinite(raw)])
    return selected


def development(track,sources):
    group,_,mapping,bundles=baseline_data(track,sources)
    keys=[(r["run"],r["frame"]) for r in group["rows"]]
    path=CACHE/f"{track}-sections.npz"
    receipt=path.with_suffix(".json")
    assert old.digest(path)==old.read(receipt)["sections_sha256"]
    with np.load(path,allow_pickle=False) as d:
        assert keys==list(zip(d["run"],d["frame"],strict=True))
        observed=d["values"][:,0]
        assert np.array_equal(d["stations"],old.STATIONS[:observed.shape[1]])
    raw=np.full((len(keys),len(old.STATIONS),2),np.nan)
    raw[:,:observed.shape[1]]=observed
    data,label_path=labels(track)
    assert keys==list(zip(data["run"],data["frame"],strict=True))
    xyz=data["lateral_m"]
    cam=np.stack((data["image_px"][:,2,:,0],data["image_px"][:,1,:,1]),axis=2)
    support=data["adjacent_invalid"]&np.isfinite(cam).all(-1)&~np.isfinite(raw)
    grouped=list(groups(raw,candidates(track,mapping,bundles,keys,raw)))
    sources.update((path,receipt,label_path,label_path.with_suffix(".json"),mapping))
    return raw,grouped,xyz,support


def score_edges(raw,grouped,truth,mask,policy):
    estimates=patch(raw,grouped,policy)
    valid=mask&np.isfinite(estimates)
    error=abs(estimates[valid])-abs(truth[valid])
    return dict(points=int(valid.sum()),mae_m=float(abs(error).mean()),signed_m=float(error.mean()),
                within_1_25_m=int((abs(error)<=1.25).sum()))


def calibrate(groups):
    stats={p:{name:score_edges(*data,p) for name,data in groups.items()} for p in TRIALS}
    baseline=stats[BASELINE]
    def metric(policy):
        rates=[stats[policy][t]["mae_m"]/baseline[t]["mae_m"] for t in DEV]
        return max(rates),sum(rates),policy
    eligible=[p for p in TRIALS if all(stats[p][t]["points"]>=baseline[t]["points"] for t in DEV)]
    choice=min(eligible,key=metric)
    spatial=min((p for p in eligible if p!=BASELINE),key=metric,default=choice)
    return choice,spatial,stats


def checks():
    grid=[np.array([0.]),np.array([0.,2.]),np.array([0.])]
    unary=[np.array([0.]),np.array([0.,1.]),np.array([0.])]
    assert solve(np.arange(3.)*2.,grid,unary,0.,BASELINE)==[0,1,0]
    assert solve(np.arange(3.)*2.,grid,unary,0.,(3.,1.))==[0,0,0]
    grid=[np.array([0.]),np.array([0.,1.]),np.array([0.,2.]),np.array([3.])]
    unary=[np.zeros(len(g)) for g in grid]
    assert solve(np.arange(4.)*2.,grid,unary,.5,(1.,1.))==[0,1,1,0]


def main():
    checks()
    cv2.setNumThreads(1)
    sources={Path(__file__),Path(candidates.__code__.co_filename),old.MOUNT,
             Path(labels.__code__.co_filename),Path(baseline_data.__code__.co_filename)}
    groups_dev={track:development(track,sources) for track in DEV}
    winner,spatial,scores=calibrate(groups_dev)
    frozen=OUT/"training.json"
    assert not any(TEST in str(p) or VAL in str(p) for p in sources)
    old.write(frozen,dict(status="frozen_before_replay",development=DEV,excluded=[TEST,VAL],seed=SEED,
              source_revision=old.revision(),winner=winner,spatial=spatial,
              trials=[dict(policy=p,metrics=v) for p,v in scores.items()],
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="continuity_selection"))
    print("Frozen",winner,spatial,scores[BASELINE],scores[spatial],flush=True)
    models,settings,pins=load_observer()
    prior=old.read(ROUTING/"straight-verification.json")
    old_side=old.read(PAIRS/"result.json")
    reports={}
    for track,name,mapping in (
            (TEST,"silverstone",CAPTURE),
            (VAL,"vallelunga",OUT.parent/"vision-vallelunga-local-corner-capture-20261112-01"/"archive-map.json")):
        directory=RAW_ARCHIVE/name
        original=directory/"sensors.npz"
        source=directory/"result.json"
        receipt=old.read(source)
        assert old.digest(original)==receipt["inputs_sha256"]
        with np.load(original,allow_pickle=False) as data:
            raw={k:data[k] for k in data.files}
        keys=list(zip(raw["run"],raw["frame"],strict=True))
        grouped=list(groups(raw["sections"],candidates(track,mapping,receipt["archive_bundles"],keys,raw["sections"],raw["motion"])))
        reference_path=ROUTING/f"{track}-state-distance.jsonl"
        reference=frames(reference_path,prior["tracks"][track]["rows_sha256"])
        modes={}
        for method,p in (("baseline",BASELINE),("spatial",spatial)):
            selected=patch(raw["sections"],grouped,p)
            sensor=OUT/f"{track}-{method}-sensors.npz"
            np.savez_compressed(sensor,**dict(raw,sections=selected))
            with np.load(sensor,allow_pickle=False) as data:
                rows,aligned,_=observe(data,models,settings)
            output=OUT/f"{track}-{method}-frames.jsonl"
            old.jsonl(output,rows)
            report,hits=metrics(rows,reference)
            result=dict(metrics=report,right_hits=hits,sensors_sha256=old.digest(sensor),frames_sha256=old.digest(output))
            if method=="baseline":
                assert result["frames_sha256"]==old_side["tracks"][track]["methods"]["baseline"]["frames_sha256"]
            if track==TEST:
                boundary=directory/"predictions/boundaries.npz"
                with np.load(boundary,allow_pickle=False) as data:
                    result["mesh"]=mesh_score(aligned,data["estimated"],reference,keys)
            modes[method]=result
        hashes={*pins,frozen,original,source,mapping,reference_path}
        if track==TEST:
            hashes.update((MESH,MESH.with_suffix(".json"),directory/"predictions/boundaries.npz"))
        reports[track]=dict(methods=modes,source_sha256={str(p):old.digest(p) for p in sorted(hashes)})
        print(track,{m:(v["right_hits"],v.get("mesh",{}).get("new_mae_m")) for m,v in modes.items()},flush=True)
    old.write(OUT/"result.json",dict(status="scored",training_sha256=old.digest(frozen),
              tracks=reports,termination="shared_geometry_side_chain_replay"))


if __name__=="__main__":
    main()
