"""Verify pixel support, archived controls and identical candidate assignments."""
from collections import Counter
from pixel_checked import (ARCHIVE,EVALUATION,METHODS,N,OUT,SILVER,WIDTH,Path,forecast_errors,frames,
                           load_labels,np,old,reference,row_map,strip_features)


def synthetic():
    image=np.full((30,60,3),[50,128,128],dtype=np.uint8)
    image[:,30:]=[120,140,130]
    camera=dict(origin_body_m=[0.,0.,0.],forward_body=[1.,0.,0.],right_body=[0.,1.,0.],
                up_body=[0.,0.,1.],projection_x=1.,projection_y=1.)
    points=np.zeros((2,7,3))
    points[:,:,0]=4.
    points[0,:,1]=np.linspace(-1.,1.,7)
    points[1,:,1]=np.linspace(9.,11.,7)
    values=strip_features(image,camera,points)
    assert values.shape==(2,N)
    assert abs(values[0,0]-70)<1.e-12 and np.isnan(values[1]).all()
    return True


def run():
    projection=synthetic()
    report=old.read(OUT/"result.json")
    train=old.read(OUT/"training.json")
    assert old.digest(OUT/"training.json")==report["training_sha256"]
    assert old.digest(OUT/"selector.joblib")==train["checkpoint_sha256"]
    for path,digest in train["source_sha256"].items():
        assert old.digest(Path(path))==digest
    control=old.read(WIDTH/"result.json")
    checked={}
    for track,folder in EVALUATION:
        r=report["tracks"][track]
        for path,digest in r["source_sha256"].items():
            assert old.digest(Path(path))==digest
        coords=OUT/f"{track}-boundaries.npz"
        assert old.digest(coords)==r["boundaries_sha256"]
        previous=WIDTH/f"{track}-boundaries.npz"
        prior=control["tracks"][track]
        assert old.digest(previous)==prior["boundaries_sha256"]
        with np.load(coords,allow_pickle=False) as d, np.load(previous,allow_pickle=False) as p:
            np.testing.assert_equal(d["width"],p["width"])
            np.testing.assert_equal(d["observed"],p["observed"])
            known=np.isfinite(d["observed"])
            np.testing.assert_equal(d["pixels"][known],d["observed"][known])
            sensor={k:d[k] for k in ("run","frame","stations")}
            boundaries={m:d[m] for m in ("observed","width","pixels")}
        archive=ARCHIVE/folder/"sensors.npz"
        with np.load(archive,allow_pickle=False) as d:
            for k,a in sensor.items():
                np.testing.assert_array_equal(a,d[k])
        labels,_=reference(track)
        keys=row_map(labels)
        modes={}
        for mode,v in r["modes"].items():
            output=frames(OUT/f"{track}-{mode}-frames.jsonl",v["trace_sha256"])
            if mode=="width":
                base=frames(WIDTH/f"{track}-width-frames.jsonl",prior["modes"]["width"]["trace_sha256"])
                for a,b in zip(output,base,strict=True):
                    assert all(a[k]==value for k,value in b.items())
            metrics=forecast_errors(output,labels)
            assert metrics==v["metrics"]
            changes=Counter()
            for a in output:
                ref=keys[a["run"],a["frame"]]
                verdict=[]
                for method in METHODS:
                    x=a["forecasts"][method]
                    code={"left":1,"right":2}.get(x["state"])
                    if code:
                        assert a["candidate"]==[code,x["distance_m"]] and 4<=x["distance_m"]<=20
                    else:
                        assert x["state"] in ("straight","abstain") and x["distance_m"] is None
                    verdict.append(old.category(ref,(code,x["distance_m"]) if code else None))
                if verdict[0]!=verdict[1]:
                    changes[f"{verdict[0]} -> {verdict[1]}"]+=1
            modes[mode]=dict(paired=changes)
        if track==SILVER:
            assert len(r["right_missing_points"])==10
            label_file=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            target,visible=load_labels(label_file,sensor,set())
            raw=boundaries["observed"]
            for mode,result in modes.items():
                added=~np.isfinite(raw)&np.isfinite(boundaries[mode])&visible
                distance=abs(boundaries[mode]-target)
                result["new_points_by_side"]={
                    side:dict(count=int(added[:,:,j].sum()),correct=int((added[:,:,j]&(distance[:,:,j]<=1.25)).sum()))
                    for j,side in enumerate(("right","left"))}
        checked[track]=modes
    old.write(OUT/"verification.json",dict(status="verified",synthetic=projection,tracks=checked,
                                          result_sha256=old.digest(OUT/"result.json")))
    print("Verified image support, unchanged observed coordinates, exact control, shared candidates and scores.",flush=True)


if __name__=="__main__":
    run()
