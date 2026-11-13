"""Verify frame pairing, preserved observations and current-image boundary additions."""
from collections import Counter
from local_transition import (EVALUATION,METHODS,OUT,SILVER,WIDTH,Path,forecast_errors,frames,np,old,reference,row_map)


def run():
    training=old.read(OUT/"training.json")
    report=old.read(OUT/"result.json")
    assert old.digest(OUT/"training.json")==report["training_sha256"]
    for path,sha in training["source_sha256"].items():
        assert old.digest(Path(path))==sha
    earlier=old.read(WIDTH/"result.json")
    verified={}
    for track,_ in EVALUATION:
        r=report["tracks"][track]
        for path,sha in r["source_sha256"].items():
            assert old.digest(Path(path))==sha
        bundle=OUT/f"{track}-boundaries.npz"
        assert old.digest(bundle)==r["boundary_sha256"]
        control=WIDTH/f"{track}-boundaries.npz"
        assert old.digest(control)==earlier["tracks"][track]["boundaries_sha256"]
        with np.load(bundle,allow_pickle=False) as d,np.load(control,allow_pickle=False) as b:
            for key in ("run","frame","stations"):
                np.testing.assert_array_equal(d[key],b[key])
            np.testing.assert_equal(d["base"],b["width"])
            np.testing.assert_equal(d["raw"],b["observed"])
            keep=np.isfinite(d["base"])
            np.testing.assert_array_equal(d["local"][keep],d["base"][keep])
            assert int((~keep&np.isfinite(d["local"])).sum())==r["extra_points"]
        truth,_=reference(track)
        keys=row_map(truth)
        evidence={}
        for mode,info in r["metrics"].items():
            rows=frames(OUT/f"{track}-{mode}-frames.jsonl",info["trace_sha256"])
            assert forecast_errors(rows,truth)==info["counts"]
            if mode=="base":
                v=earlier["tracks"][track]["modes"]["width"]
                old_rows=frames(WIDTH/f"{track}-width-frames.jsonl",v["trace_sha256"])
                for a,b in zip(rows,old_rows,strict=True):
                    assert all(a[k]==p for k,p in b.items())
            differences=Counter()
            for row in rows:
                pair=[]
                label=keys[row["run"],row["frame"]]
                for method in METHODS:
                    forecast=row["forecasts"][method]
                    state=forecast["state"]
                    dist=forecast["distance_m"]
                    code={"left":1,"right":2}.get(state)
                    if code:
                        assert 4<=dist<=20 and row["candidate"]==[code,dist]
                    else:
                        assert state in ("abstain","straight") and dist is None
                    pair.append(old.category(label,(code,dist) if code else None))
                if pair[0]!=pair[1]:
                    differences[f"{pair[0]} -> {pair[1]}"]+=1
            evidence[mode]=differences
        if track==SILVER:
            assert all(abs(a["shift_m"])<=.75 for rows in r["local_diagnostics"].values() for a in rows)
        verified[track]=evidence
    old.write(OUT/"verification.json",dict(status="verified",tracks=verified,result_sha256=old.digest(OUT/"result.json")))
    print("Verified archive and model hashes, exact control, preserved observations, and shared forecast offers.",flush=True)


if __name__=="__main__":
    run()
