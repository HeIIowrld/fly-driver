"""Verify the shared phase guard and exact replay controls."""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import json

import numpy as np

from phase_features import OUT, PARENT, SILVER, VAL, current_features, frame_features, old
from evaluate_phase import POLICIES, OUTPUTS
from rank import METHODS

ROOT = OUT.parents[1]


def digest(path):
    with path.open("rb") as file:
        return file_digest(file,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def check_causality():
    frames = [dict(run="synthetic",frame=i,generation="test",received_s=i*.1) for i in range(5)]
    station = old.STATIONS
    base = np.stack((-3.+station**2*.003,3.+station**2*.003),axis=-1)
    sections = np.stack([base+i*.1 for i in range(5)])
    motion = np.asarray([[i*.1,15.,.1+i*.02] for i in range(5)])
    full = frame_features(frames,sections,motion)
    prefix = frame_features(frames[:3],sections[:3],motion[:3])
    np.testing.assert_allclose(full[:3],prefix)
    opposite = motion[0].copy()
    opposite[2] *= -1
    np.testing.assert_allclose(current_features(sections[0],motion[0]),current_features(-sections[0,:,::-1],opposite))
    changed = [*frames[:3],dict(frames[3],generation="new"),dict(frames[4],generation="new")]
    reset = frame_features(changed,sections,motion)
    assert reset[3,24]==0 and reset[4,24]==1 and full[3,24]==3


def kind(row,offer):
    if offer is None:
        return "quiet"
    if row["truth"]!="approach":
        return row["truth"]+"_alert"
    if str(offer[0])!=row["label"]:
        return "wrong_direction"
    return "joint_hit" if max(abs(offer[1]-v) for v in row["truth_endpoints_m"])<=5. else "distance_error"


def validate_hashes(table, excluded=False):
    for name,sha in table.items():
        if excluded:
            assert SILVER not in name and VAL not in name
        assert digest(ROOT/name)==sha,name


def check_training(path,model):
    training = read(path)
    assert set(training["excluded"])=={SILVER,VAL}
    assert not set(training["development"])&{SILVER,VAL}
    assert digest(model)==training["checkpoint_sha256"]
    validate_hashes(training["source_sha256"],excluded=True)
    return training


def main():
    check_causality()
    training_path, result_path = OUT/"training.json",OUT/"result.json"
    trained = check_training(training_path,OUT/"phase.joblib")
    result = read(result_path)
    assert trained["cuts"]==result["cuts"]
    assert trained["veto_corner_probability_max"]==result["phase_veto"]
    validate_hashes(result["source_sha256"])
    parent = read(PARENT/"result.json")
    report = {}
    for track in (SILVER,VAL):
        path,control = OUT/f"{track}-frames.jsonl",PARENT/f"{track}-frames.jsonl"
        assert digest(path)==result["tracks"][track]["rows_sha256"]
        assert digest(control)==parent["tracks"][track]["rows_sha256"]
        rows,original = lines(path),lines(control)
        assert len(rows)==len(original)==1200
        counted,correct = {m:Counter() for m in METHODS},[]
        for row,before in zip(rows,original,strict=True):
            assert (row["run"],row["frame"],row["truth"],row["label"])==(before["run"],before["frame"],before["truth"],before["label"])
            assert row["scores"]==before["scores"]
            assert row["scores"]["temporal"]["candidate"]==row["scores"]["t4t5_confirm"]["candidate"]
            phase = row["phase"]
            if phase is not None:
                np.testing.assert_allclose(sum(phase),1.,atol=1e-6)
            for m in METHODS:
                a = row["predictions"][f"{m}_original_candidate_only"]
                b = row["predictions"][f"{m}_veto_candidate_only"]
                assert a==before["final"][f"{m}_candidate_only"] and (b is None or b==a)
                if a is not None:
                    assert (b is not None)==(phase[2]<=trained["veto_corner_probability_max"])
                    if b is None:
                        counted[m][kind(row,a)] += 1
                    if kind(row,a)=="joint_hit":
                        correct.append(dict(method=m,frame=row["frame"],label=row["label"],offer=a,
                                            kept=b is not None,inside_probability=phase[2]))
                for p in POLICIES:
                    selected = row["predictions"][f"{m}_{p}_candidate_only"]
                    assert selected is None or selected==row["scores"][m]["candidate"]
                    assert row["predictions"][f"{m}_{p}_shared_edge"]==(row["common_edge"] or selected)
                    for base in OUTPUTS:
                        assert row["states"][f"{m}_{p}_{base}"] in {"left","right","straight","abstain"}
        report[track] = dict(veto_removed={m:dict(c) for m,c in counted.items()},correct=correct)

    direct = OUT/"direct-phase"
    d_manifest,d_result = direct/"training.json",direct/"result.json"
    d = check_training(d_manifest,direct/"rankers.joblib")
    test = read(d_result)
    assert digest(d_manifest)==test["training_sha256"] and d["cuts"]==test["cuts"]
    for track,g in test["tracks"].items():
        validate_hashes(g["source_sha256"])
        path = direct/f"{track}-frames.jsonl"
        assert digest(path)==g["rows_sha256"]
        records,prior = lines(path),lines(PARENT/f"{track}-frames.jsonl")
        assert len(records)==len(prior)==1200
        for r,b in zip(records,prior,strict=True):
            assert (r["run"],r["frame"],r["pool"])==(b["run"],b["frame"],b["pool"])
            assert r["scores"]["temporal"]["candidate"]==r["scores"]["t4t5_confirm"]["candidate"]
            for m in METHODS:
                item = r["scores"][m]
                selected = item["candidate"] if item["score"] is not None and item["score"]>=d["cuts"][m] else None
                assert selected==r["final"][f"{m}_candidate_only"]
    evidence = dict(status="verified",seed=trained["seed"],source_revision=trained["source_revision"],
                    training_sha256=digest(training_path),result_sha256=digest(result_path),
                    direct_training_sha256=digest(d_manifest),direct_result_sha256=digest(d_result),
                    tracks=report,verifier_sha256=digest(Path(__file__)),
                    termination="phase_causality_and_identical_candidate_replays_verified")
    (OUT/"verification.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
