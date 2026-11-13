"""Verify the alternate cost policy and paired replay."""
from collections import Counter
from compare_equal_cost import EVALUATION,METHODS,OUT,OUTPUT,forecast_errors,np,old,reference,row_map
from local_transition import frames,Path,SILVER,load_labels


def run():
    result=old.read(OUTPUT/"result.json")
    frozen=old.read(OUTPUT/"training.json")
    training=old.read(OUT/"training.json")
    assert old.digest(OUTPUT/"training.json")==result["training_sha256"]
    assert old.digest(OUT/"training.json")==frozen["prior_training_sha256"]
    choices=[(mode,float(c),r) for mode,points in training["totals"].items() for c,r in points.items()]
    mode,cut,stats=max(choices,key=lambda a:(2*a[2]["correct"]-a[2]["accepted"],a[2]["correct"],a[1]))
    assert (mode,cut,stats)==(frozen["mode"],frozen["cut"],frozen["development"])
    summary={}
    for track,_ in EVALUATION:
        receipt=result["tracks"][track]
        for name,digest in receipt["source_sha256"].items():
            assert old.digest(Path(name))==digest
        changed=OUTPUT/f"{track}-boundaries.npz"
        assert old.digest(changed)==receipt["boundaries_sha256"]
        with np.load(changed,allow_pickle=False) as d,np.load(OUT/f"{track}-boundaries.npz",allow_pickle=False) as base:
            for k in ("run","frame","stations"):
                np.testing.assert_array_equal(d[k],base[k])
            available=np.isfinite(base["base"])
            np.testing.assert_array_equal(d["sections"][available],base["base"][available])
            extra=~available&np.isfinite(d["sections"])
            assert int(extra.sum())==receipt["added"]
            geometry={}
            if track==SILVER:
                mesh=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
                target,valid=load_labels(mesh,{key:d[key] for key in ("run","frame","stations")},set())
                for j,side in enumerate(("right","left")):
                    chosen=extra[:,:,j]&valid[:,:,j]
                    errors=abs(d["sections"][:,:,j][chosen]-target[:,:,j][chosen])
                    geometry[side]=dict(labeled=int(chosen.sum()),within_125cm=int((errors<=1.25).sum()))
        label,_=reference(track)
        truth=row_map(label)
        rows=frames(OUTPUT/f"{track}-frames.jsonl",receipt["trace_sha256"])
        assert forecast_errors(rows,label)==receipt["metrics"]
        pair=Counter()
        for row in rows:
            ref=truth[row["run"],row["frame"]]
            states=[]
            for method in METHODS:
                x=row["forecasts"][method]
                side={"left":1,"right":2}.get(x["state"])
                if side:
                    assert 4<=x["distance_m"]<=20 and row["candidate"]==[side,x["distance_m"]]
                else:
                    assert x["state"] in ("abstain","straight") and x["distance_m"] is None
                states.append(old.category(ref,(side,x["distance_m"]) if side else None))
            if states[0]!=states[1]:
                pair[f"{states[0]} -> {states[1]}"]+=1
        summary[track]=dict(paired=pair,geometry=geometry)
    old.write(OUTPUT/"verification.json",dict(status="verified",tracks=summary,result_sha256=old.digest(OUTPUT/"result.json")))
    print("Verified equal-cost selection, preserved coordinates, and the paired replay.",flush=True)


if __name__=="__main__":
    run()
