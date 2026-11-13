"""Verify temporal causality, candidate identity, and frozen thresholds."""
from hashlib import file_digest
from pathlib import Path
from collections import Counter
import json

import numpy as np

from alternatives import history_vector, transform, advance, OUT, SILVER, VAL
from rank import METHODS

ROOT = OUT.parents[1]


def digest(path):
    with path.open("rb") as file:
        return file_digest(file,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def causal_check():
    previous = dict(offer=(2,15.),point=np.array([15.,-.8]))
    motions = ([1.,40.,.18],[1.1,39.,.18],[1.2,38.,.2])
    steps = [transform(b,a) for a,b in zip(motions,motions[1:])]
    point = advance(advance(previous["point"],*steps[0]),*steps[1])
    current = dict(offer=(2,point[0]),point=point)
    features = history_vector(current,[[],[previous],[]],[None,steps,None])
    np.testing.assert_allclose(features,[0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,1.],atol=1e-8)
    assert history_vector(current,[[],[previous],[]],[None,None,None]) == [0.]*13


def category(row,candidate):
    if candidate is None:
        return "quiet"
    if row["truth"]!="approach":
        return row["truth"]+"_alert"
    direction,distance = candidate
    if str(direction)!=row["label"]:
        return "wrong_direction"
    return "joint_hit" if max(abs(distance-v) for v in row["truth_endpoints_m"])<=5. else "distance_error"


def main():
    causal_check()
    training_path, result_path = OUT/"training.json",OUT/"result.json"
    training,result = read(training_path),read(result_path)
    assert digest(training_path)==result["training_sha256"]
    assert training["cuts"]==result["cuts"]
    assert training["excluded"]==[SILVER,VAL] and not set(training["development"])&{SILVER,VAL}
    assert digest(OUT/"rankers.joblib")==training["checkpoint_sha256"]
    for name,sha in training["source_sha256"].items():
        assert not any(t in name for t in (SILVER,VAL)) and digest(ROOT/name)==sha,name
    verification = {}
    for track,report in result["tracks"].items():
        for name,sha in report["source_sha256"].items():
            assert digest(ROOT/name)==sha,name
        path = OUT/f"{track}-frames.jsonl"
        assert digest(path)==report["rows_sha256"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
        keys = [(r["run"],r["frame"]) for r in rows]
        assert len(keys)==len(set(keys))==1200
        obs = OUT/f"{track}-observations.npz"
        assert digest(obs)==read(obs.with_suffix(".json"))["arrays_sha256"]
        with np.load(obs,allow_pickle=False) as source:
            assert list(zip(source["run"],source["frame"],strict=True))==keys
            assert set(source.files)=={"run","frame","motion","sections","primary","edge_base","heads"}
            assert source["motion"].shape==(1200,3)
        comparison, new_right, temporal_changes = Counter(),[],[]
        for row in rows:
            if row["primary"] is not None:
                assert row["primary"] in row["pool"]
            assert row["scores"]["temporal"]["candidate"]==row["scores"]["t4t5_confirm"]["candidate"]
            for o in row["pool"]:
                assert o[0] in (1,2) and 4.<=o[1]<=20.
            for m in METHODS:
                score = row["scores"][m]
                candidate = score["candidate"]
                if candidate:
                    assert candidate in row["pool"]
                accept = candidate if score["score"] is not None and score["score"]>=training["cuts"][m] else None
                assert row["final"][f"{m}_candidate_only"]==accept
                assert row["final"][f"{m}_shared_edge"]==(row["common_edge"] or accept)
                assert row["states"][f"{m}_candidate_only"] in {"straight","left","right","abstain"}
            if row["truth"]=="approach" and row["label"]=="2":
                a,b = (category(row,row["scores"][m]["candidate"]) for m in ("static","temporal"))
                comparison[a,b] += 1
                if a!=b:
                    temporal_changes.append(dict(frame=row["frame"],before=a,after=b,
                                                 static=row["scores"]["static"],temporal=row["scores"]["temporal"]))
                if category(row,row["primary"])!="joint_hit" and any(category(row,o)=="joint_hit" for o in row["pool"]):
                    new_right.append(row["frame"])
        verification[track] = dict(right_candidate_ranking_changes=temporal_changes,
                                   right_pool_new_frames=new_right,
                                   static_to_temporal=[dict(before=k[0],after=k[1],frames=v)
                                                       for k,v in sorted(comparison.items())])
    summary = dict(status="verified",source_revision=training["source_revision"],seed=training["seed"],
                   training_sha256=digest(training_path),result_sha256=digest(result_path),
                   verifier_sha256=digest(Path(__file__)),tracks=verification,
                   termination="two_timelines_temporal_step_and_paired_selection_verified")
    (OUT/"verification.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":
    main()
