"""Check boundary reproduction, causal donor support, and paired forecast contracts."""
from collections import Counter
from width_conditioned import (ARCHIVE,EVALUATION,METHODS,OUT,PREV,SILVER,Path,SensorArrays,apply,bank,choose,
                               frames,joblib,load_labels,load_observer,np,old,quality)


def run():
    result=old.read(OUT/"result.json")
    trained=old.read(OUT/"training.json")
    assert old.digest(OUT/"training.json")==result["training_sha256"]
    assert old.digest(OUT/"selector.joblib")==trained["checkpoint_sha256"]
    for source,digest in trained["source_sha256"].items():
        assert old.digest(Path(source))==digest
    policy=trained["totals"][str(trained["threshold"])]
    assert policy["correct"]>=trained["precision_floor"]*policy["accepted"]
    model=joblib.load(OUT/"selector.joblib")
    _,settings,_=load_observer()
    receipt={}
    for track,folder in EVALUATION:
        info=result["tracks"][track]
        for source,digest in info["source_sha256"].items():
            assert old.digest(Path(source))==digest
        coords=OUT/f"{track}-boundaries.npz"
        assert old.digest(coords)==info["boundaries_sha256"]
        with np.load(ARCHIVE/folder/"sensors.npz",allow_pickle=False) as d:
            sensors=SensorArrays({k:d[k] for k in d.files})
        with np.load(coords,allow_pickle=False) as d:
            for k in ("run","frame","stations"):
                np.testing.assert_array_equal(d[k],sensors[k])
            raw=sensors["sections"]
            np.testing.assert_equal(d["observed"],raw)
            modes={key:d[key] for key in ("prior","width")}
        observed=np.isfinite(raw)
        audit={}
        for key,edges in modes.items():
            np.testing.assert_array_equal(edges[observed],raw[observed])
            count=int((~observed&np.isfinite(edges)).sum())
            assert count==info["modes"][key]["points"]
            rows=frames(OUT/f"{track}-{key}-frames.jsonl",info["modes"][key]["trace_sha256"])
            states={m:Counter() for m in METHODS}
            for r in rows:
                for m,x in r["forecasts"].items():
                    state,d=x["state"],x["distance_m"]
                    if state in ("left","right"):
                        assert 4<=d<=20
                        assert r["candidate"]==[{"left":1,"right":2}[state],d]
                    else:
                        assert state in ("straight","abstain") and d is None
                    states[m][state]+=1
            audit[key]=dict(points=count,states=states)
        if track==SILVER:
            n=812
            prefix=SensorArrays({k:v if k=="stations" else v[:n] for k,v in sensors.items()})
            options=bank(prefix,settings)
            patched=apply(prefix["sections"],options,choose(options,model),trained["threshold"])
            np.testing.assert_allclose(patched,modes["width"][:n],rtol=0.,atol=1.e-12)
            mesh=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            truth,visible=load_labels(mesh,sensors,set())
            for key,edges in modes.items():
                audit[key]["silverstone_labeled"]={
                    side:quality(raw[:,:,j],edges[:,:,j],truth[:,:,j],visible[:,:,j])
                    for j,side in enumerate(("right","left"))
                }
            whole=bank(sensors,settings)
            np.testing.assert_allclose(apply(raw,whole,choose(whole,model),trained["threshold"]),modes["width"],
                                       rtol=0.,atol=1.e-12)
        receipt[track]=audit
    old.write(OUT/"verification.json",dict(status="verified",tracks=receipt,result_sha256=old.digest(OUT/"result.json")))
    print("Verified fixed precision rule, unchanged observations, shared candidate states and causal prefix.",flush=True)


if __name__=="__main__":
    run()
