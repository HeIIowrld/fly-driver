"""Verify the continuous trace solver and diagnose causal support on missing right edges."""
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from trace_chains import OUT, PAIRS, TEST, VAL, DEV, RAW_ARCHIVE, ROUTING, old, frames, solve, huber
from replay_corner_boundary_states import load_observer, sensor_rows, alt
from aligned_sections import project_sections
from verify_side import classify

METHODS=("temporal","t4t5_confirm")


def verify_solver():
    rng=np.random.default_rng(17)
    for _ in range(25):
        stations=np.cumsum(rng.integers(1,5,size=5)).astype(float)
        grids=[np.sort(rng.normal(size=3)) for _ in stations]
        scores=[rng.normal(size=3) for _ in stations]
        prior=float(rng.normal())
        weights=(3.,1.)
        path=solve(stations,grids,scores,prior,weights)
        def cost(seq):
            points=[grid[i] for grid,i in zip(grids,seq,strict=True)]
            slope=np.diff(points)/np.diff(stations)
            return (sum(s[i] for s,i in zip(scores,seq,strict=True))
                    -weights[0]*huber(slope-prior).sum()-weights[1]*huber(np.diff(slope)).sum())
        np.testing.assert_allclose(cost(path),max(cost(seq) for seq in product(range(3),repeat=5)),atol=1.e-11)


def projected_side(current, i, rows, motion, raw, side, stations):
    latest=i
    for previous in range(i-1,max(i-5,-1),-1):
        if (not alt.adjacent(rows[previous],rows[previous+1])
                or not np.isfinite(motion[previous:previous+2]).all()
                or min(motion[previous:previous+2,1])<old.MIN_SPEED_MPS):
            break
        age=motion[latest,0]-motion[previous,0]
        if age>.45:
            break
        chain=[alt.transform(motion[k+1],motion[k]) for k in range(previous,latest)]
        if any(t is None for t in chain):
            break
        available=np.isfinite(raw[previous,:,side])
        if not available.any():
            continue
        points=np.column_stack((stations[available],raw[previous,available,side]))
        for step in chain:
            points=alt.advance(points,*step)
        y=project_sections(stations,np.repeat(points[:,None],2,axis=1))[:,0]
        same=np.isfinite(current)&np.isfinite(y)
        yield rows[previous]["frame"],y,same


def temporal_diagnostic(path,settings):
    audit=OUT.parent/"vision-corner-outward-camera-20261112-01"/"right-camera-audit.json"
    points=old.read(audit)["points"]
    with np.load(path,allow_pickle=False) as data:
        rows=sensor_rows(data,settings)
        motion,raw,stations=(data[k] for k in ("motion","sections","stations"))
    by_frame={row["frame"]:i for i,row in enumerate(rows)}
    found=[]
    for point in points:
        frame,target,physical=(point[k] for k in ("frame","station_m","lateral_m"))
        i=by_frame[frame]
        s=int(np.flatnonzero(stations==target)[0])
        current=raw[i,:,0]
        history=[]
        for old_frame,projected,overlap in projected_side(current,i,rows,motion,raw,0,stations):
            if not np.isfinite(projected[s]):
                continue
            nearby=overlap&(abs(stations-target)<=6.)
            history.append(dict(frame=old_frame,projected_m=float(projected[s]),
                                absolute_error_m=float(abs(projected[s]-physical)),
                                anchored=bool(nearby.any() and max(abs(projected[nearby]-current[nearby]))<=1.25)))
        found.append(dict(frame=frame,station_m=target,history=history))
    n=sum(any(p["anchored"] and p["absolute_error_m"]<=1.25 for p in row["history"]) for row in found)
    return dict(points=found,points_with_accurate_anchored_prior=n,
                source_sha256={str(p):old.digest(p) for p in (path,audit)})


def main():
    verify_solver()
    result_path=OUT/"result.json"
    training_path=OUT/"training.json"
    report=old.read(result_path)
    training=old.read(training_path)
    assert old.digest(training_path)==report["training_sha256"]
    assert training["development"]==list(DEV) and training["excluded"]==[TEST,VAL]
    for name,sha in training["source_sha256"].items():
        assert TEST not in name and VAL not in name and old.digest(Path(name))==sha
    prior=old.read(ROUTING/"straight-verification.json")
    result={}
    for track,entry in report["tracks"].items():
        for name,sha in entry["source_sha256"].items():
            assert old.digest(Path(name))==sha
        reference=frames(ROUTING/f"{track}-state-distance.jsonl",prior["tracks"][track]["rows_sha256"])
        by_mode={}
        for name,m in entry["methods"].items():
            path=OUT/f"{track}-{name}-sensors.npz"
            assert old.digest(path)==m["sensors_sha256"]
            replay=frames(OUT/f"{track}-{name}-frames.jsonl",m["frames_sha256"])
            assert len(replay)==len(reference)==1200
            paired=Counter()
            for r,t in zip(replay,reference,strict=True):
                assert (r["run"],r["frame"])==(t["run"],t["frame"])
                a,b=(classify(r,t,mode) for mode in METHODS)
                if a!=b:
                    paired[t["truth"],t["label"],a,b]+=1
            by_mode[name]=[dict(truth=k[0],label=k[1],edge=k[2],motion=k[3],frames=n)
                           for k,n in sorted(paired.items())]
        result[track]=by_mode
    _,settings,_=load_observer()
    temporal=temporal_diagnostic(RAW_ARCHIVE/"silverstone/sensors.npz",settings)
    old.write(OUT/"verification.json",dict(status="verified",solver_exhaustive_checks=25,paired=result,
              temporal_diagnostic=temporal,result_sha256=old.digest(result_path),termination="archive_scores_verified"))
    print("Accurate anchored prior",temporal["points_with_accurate_anchored_prior"],flush=True)


if __name__=="__main__":
    main()
