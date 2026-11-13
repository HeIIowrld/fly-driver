"""Replay the policy chosen with equal costs for missed and inaccurate new edge points."""
from local_transition import (ARCHIVE,EVALUATION,METHODS,OUT,SILVER,SensorArrays,add,bank,boundary_scores,
                              forecast_errors,load_labels,load_observer,np,observe,old,reference,row_map,search)

OUTPUT=OUT/"equal-cost"


def run():
    OUTPUT.mkdir(exist_ok=True)
    source=OUT/"training.json"
    training=old.read(source)
    options=[(mode,float(c),r) for mode,cuts in training["totals"].items() for c,r in cuts.items()]
    mode,cut,stats=max(options,key=lambda a:(2*a[2]["correct"]-a[2]["accepted"],a[2]["correct"],a[1]))
    frozen=OUTPUT/"training.json"
    old.write(frozen,dict(mode=mode,cut=cut,development=stats,prior_training_sha256=old.digest(source),
                          source_sha256=old.digest(OUT/"compare_equal_cost.py"),selection="equal_cost_development_only"))
    print("Equal-cost policy",mode,cut,stats,flush=True)
    observer,settings,_=load_observer()
    results={}
    for track,folder in EVALUATION:
        root=ARCHIVE/folder
        metadata=root/"result.json"
        proof=old.read(metadata)
        archive=root/"sensors.npz"
        assert old.digest(archive)==proof["inputs_sha256"]
        with np.load(archive,allow_pickle=False) as d:
            sensor=SensorArrays({k:d[k] for k in d.files})
        paths={archive,metadata}
        original=bank(sensor,settings)
        search_result=search(track,sensor,original,proof["archive_bundles"],paths)[mode]
        calibrated=OUT/f"{track}-boundaries.npz"
        with np.load(calibrated,allow_pickle=False) as d:
            np.testing.assert_array_equal(d["run"],sensor["run"])
            np.testing.assert_array_equal(d["frame"],sensor["frame"])
            base=d["base"]
        paths.add(calibrated)
        raw=sensor["sections"]
        repaired=add(raw,base,search_result,cut)
        truth,path=reference(track)
        paths.add(path)
        refs=row_map(truth)
        prediction=observe(SensorArrays(dict(sensor,sections=repaired)),observer,settings)[0]
        scored=[r for r in prediction if (r["run"],r["frame"]) in refs]
        trace=OUTPUT/f"{track}-frames.jsonl"
        old.jsonl(trace,scored)
        result=dict(metrics=forecast_errors(scored,truth),trace_sha256=old.digest(trace),
                    added=int((~np.isfinite(base)&np.isfinite(repaired)).sum()))
        if track==SILVER:
            mesh=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge,valid=load_labels(mesh,sensor,paths)
            right=np.array([refs[r,f]["truth"]=="approach" and refs[r,f]["label"]=="2"
                            for r,f in zip(sensor["run"],sensor["frame"],strict=True)])
            assert right.sum()==13
            result["right_approach"]={side:boundary_scores(raw[right,:,j],repaired[right,:,j],
                edge[right,:,j],valid[right,:,j]) for j,side in enumerate(("right","left"))}
            additions=[]
            for i,j,k in np.argwhere(right[:,None,None]&~np.isfinite(base)&np.isfinite(repaired)&valid):
                additions.append(dict(frame=int(sensor["frame"][i]),station_m=float(sensor["stations"][j]),
                    side=("right","left")[k],error_m=float(abs(repaired[i,j,k]-edge[i,j,k]))))
            result["right_approach_additions"]=additions
        coordinates=OUTPUT/f"{track}-boundaries.npz"
        np.savez_compressed(coordinates,run=sensor["run"],frame=sensor["frame"],stations=sensor["stations"],
                            sections=repaired)
        result["boundaries_sha256"]=old.digest(coordinates)
        result["source_sha256"]={str(p):old.digest(p) for p in sorted(paths)}
        results[track]=result
        print(track,result["added"],result.get("right_approach_additions"),result["metrics"],flush=True)
    old.write(OUTPUT/"result.json",dict(tracks=results,training_sha256=old.digest(frozen),termination="paired_scored"))


if __name__=="__main__":
    run()
