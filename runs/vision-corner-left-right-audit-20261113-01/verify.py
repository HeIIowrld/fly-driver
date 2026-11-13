"""Verify paired replay, failure counts, and unchanged right-turn forecasts."""
from collections import Counter
from audit_and_refit import CLI, EVAL, METHODS, OUT, Path, TURNS, frames, label, old, reference, score_archive


def index(rows):
    return {(r["run"],r["frame"]):r for r in rows}


def run():
    audit=old.read(OUT/"result.json")
    test=old.read(OUT/"raw-left-supplement-result.json")
    training=OUT/"training.json"
    assert old.digest(training)==audit["training_sha256"]
    assert old.digest(OUT/"alternative-confirmers.joblib")==old.read(training)["checkpoint_sha256"]
    proof={}
    for track,name in EVAL:
        root=CLI/("vallelunga-retry" if name=="vallelunga" else name)
        base=root/"predictions/forecasts.jsonl"
        before=frames(base,old.read(base.with_name("result.json"))["forecasts_sha256"])
        trial=test["tracks"][track]
        after=frames(OUT/f"{track}-raw-left-supplement.jsonl",trial["forecasts_sha256"])
        for source,sha in trial["source_sha256"].items():
            assert old.digest(Path(source))==sha
        detail=frames(OUT/f"{track}-audit.jsonl",audit["tracks"][track]["audit_sha256"])
        labels,_=reference(track)
        keys=index(labels)
        assert len(detail)==sum(r["truth"]=="approach" for r in labels)
        for side in (1,2):
            assert Counter(r["reason"] for r in detail if r["side"]==side)==audit["tracks"][track]["issues"][str(side)]
        hits=[]
        for a,b in zip(before,after,strict=True):
            assert (a["run"],a["frame"])==(b["run"],b["frame"])
            for method in METHODS:
                old_f,new_f=a["forecasts"][method],b["forecasts"][method]
                if old_f["state"] in TURNS:
                    assert a==b
                if old_f==new_f:
                    continue
                assert new_f["state"]=="left" and b["candidate"]==[1,new_f["distance_m"]]
                ref=keys.get((b["run"],b["frame"]))
                if ref is not None and label(ref,1,new_f["distance_m"])=="hit_1":
                    hits.append(dict(method=method,frame=b["frame"],prediction_m=new_f["distance_m"],
                        reference_m=ref["truth_endpoints_m"],
                        max_error_m=max(abs(new_f["distance_m"]-x) for x in ref["truth_endpoints_m"])))
        assert all(score_archive(labels,index(after),m)==trial["scores"]["after"][m] for m in METHODS)
        proof[track]=dict(new_left_hits=hits,frames=len(after))
        print(track,hits,flush=True)
    old.write(OUT/"verification.json",dict(status="verified",tracks=proof,
        inputs={str(OUT/f):old.digest(OUT/f) for f in ("result.json","raw-left-supplement-result.json")}))


if __name__=="__main__":
    run()
