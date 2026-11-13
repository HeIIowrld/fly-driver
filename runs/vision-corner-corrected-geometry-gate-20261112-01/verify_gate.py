"""Audit new confirmation thresholds on real side geometry and the offline intervention."""
from pathlib import Path

import numpy as np

from train_gate import OUT,ARCHIVE,SILVER,VAL,MONZA,KINDS,MODES,old,load_observer,reference,frames,joblib,replay

SIDE=OUT.parent/"vision-corner-outward-camera-20261112-01"
ORACLE=OUT.parent/"vision-corner-side-three-tracks-20261112-01"


def summarize(rows):
    result={}
    for name in rows[0]["decisions"]:
        subset=[r for r in rows if r["truth"]=="approach" and r["label"]=="2"]
        accepted=[r for r in subset if r["decisions"][name]["category"]=="joint_hit"]
        result[name]=dict(right_approach=len(subset),right_joint=len(accepted),
                          right_joint_frames=[r["frame"] for r in accepted])
    return result


def main():
    trained=OUT/"training.json"
    result_path=OUT/"result.json"
    training,result=old.read(trained),old.read(result_path)
    assert old.digest(trained)==result["training_sha256"]
    assert training["excluded"]==[SILVER,VAL,MONZA]
    weights=OUT/"confirmers.joblib"
    assert old.digest(weights)==training["checkpoint_sha256"]
    for name,sha in training["source_sha256"].items():
        assert all(t not in name for t in training["excluded"])
        assert old.digest(Path(name))==sha
    for track,record in result["tracks"].items():
        for name,sha in record["source_sha256"].items():
            assert old.digest(Path(name))==sha
        rows=frames(OUT/f"{track}-frames.jsonl",record["rows_sha256"])
        assert len(rows)==record["frames"]
        expected=reference(track)[0]
        assert {(r["run"],r["frame"]) for r in rows}=={(r["run"],r["frame"]) for r in expected}
    models,settings,sources=load_observer()
    heads=joblib.load(weights)
    side_report=SIDE/"side-result.json"
    oracle_report=ORACLE/"geometry-intervention.json"
    side=old.read(side_report)
    oracle=old.read(oracle_report)
    diagnostics={}
    for track in (SILVER,VAL):
        truth,ref=reference(track)
        sources.add(ref)
        path=SIDE/f"{track}-normal_anchor-sensors.npz"
        assert old.digest(path)==side["tracks"][track]["methods"]["normal_anchor"]["sensor_sha256"]
        cases={"side_contrast":path}
        if track==SILVER:
            for mode in ("fill_right","replace_right","replace_both"):
                path=ORACLE/f"oracle-{mode}-sensors.npz"
                assert old.digest(path)==oracle["tracks"][mode]["sensor_sha256"]
                cases[mode]=path
        diagnostics[track]={}
        for mode,path in cases.items():
            rows,metrics=replay(path,truth,None,models,settings,heads,training)
            result_file=OUT/f"{track}-{mode}-frames.jsonl"
            old.jsonl(result_file,rows)
            diagnostics[track][mode]=dict(metrics=metrics,right=summarize(rows),
                rows_sha256=old.digest(result_file),inputs_sha256=old.digest(path))
            sources.add(path)
    sources.update((Path(__file__),result_path,trained,weights,side_report,oracle_report))
    old.write(OUT/"verification.json",dict(status="verified",three_raw_archives=True,diagnostics=diagnostics,
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="heldout_and_oracle_cases_audited"))
    print({t:{m:r["right"] for m,r in cases.items()} for t,cases in diagnostics.items()},flush=True)


if __name__=="__main__":
    main()
