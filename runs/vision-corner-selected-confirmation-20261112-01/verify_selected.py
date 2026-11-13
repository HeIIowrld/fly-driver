"""Verify nested-selection provenance and paired changes at the frozen thresholds."""
from collections import Counter
from pathlib import Path

from selected_gate import OUT,GATE,DEVELOP,SILVER,VAL,MONZA,MODES,KINDS,old,frames,joblib


def main():
    training_path,result_path=OUT/"training.json",OUT/"result.json"
    training,result=old.read(training_path),old.read(result_path)
    assert old.digest(training_path)==result["training_sha256"]
    assert training["development"]==list(DEVELOP) and training["excluded"]==[SILVER,VAL,MONZA]
    checkpoint=OUT/"confirmers.joblib"
    assert old.digest(checkpoint)==training["checkpoint_sha256"]
    models=joblib.load(checkpoint)
    expected={"joint_hit","wrong_direction","distance_error","outside_alert","in_corner_alert"}
    for model in models["structured"].values():
        assert set(model["model"].estimator.classes_)==expected
    for held,fold in training["folds"].items():
        assert set(fold["inner"])==set(DEVELOP)-{held}
    for name,sha in training["source_sha256"].items():
        assert all(track not in name for track in training["excluded"]) and old.digest(Path(name))==sha
    for values in training["development_metrics"].values():
        for m in values.values():
            assert m["outside"]<=training["caps"]["outside"] and m["inside"]<=training["caps"]["inside"]
    original=old.read(GATE/"result.json")
    evidence={}
    for track,item in result["tracks"].items():
        for name,sha in item["source_sha256"].items():
            assert old.digest(Path(name))==sha
        before=frames(GATE/f"{track}-frames.jsonl",original["tracks"][track]["rows_sha256"])
        after=frames(OUT/f"{track}-frames.jsonl",item["rows_sha256"])
        assert len(before)==len(after)==item["frames"]
        report={m:Counter() for m in MODES}
        changed=[]
        for a,b in zip(before,after,strict=True):
            for key in ("run","frame","truth","label","candidate"):
                assert a[key]==b[key]
            for kind in KINDS:
                for method in ("fixed","observed"):
                    assert a["decisions"][f"{method}_{kind}"]==b["decisions"][f"{method}_{kind}"]
            differences={}
            for m in MODES:
                x,y=(b["decisions"][f"{m}_{kind}"]["category"] for kind in KINDS)
                if x!=y:
                    report[m][b["truth"],b["label"],x,y]+=1
                    differences[m]=(x,y)
            if differences:
                changed.append(dict(run=b["run"],frame=b["frame"],truth=b["truth"],label=b["label"],
                                    candidate=b["candidate"],transitions=differences))
        trace=OUT/f"{track}-paired.jsonl"
        old.jsonl(trace,changed)
        evidence[track]=dict(rows_sha256=old.digest(trace),
            transitions={m:[dict(truth=k[0],label=k[1],edge=k[2],motion=k[3],frames=n)
                           for k,n in sorted(c.items())] for m,c in report.items()})
    old.write(OUT/"verification.json",dict(status="verified",result_sha256=old.digest(result_path),tracks=evidence,
              termination="three_replays_share_all_candidates_and_controls"))


if __name__=="__main__":
    main()
