"""Replay a shared phase detector without changing the candidate selected in either condition."""
from pathlib import Path
from collections import Counter, defaultdict
import json

import numpy as np

from train_phase import OUT, PARENT, DEVELOP, SILVER, VAL, group, phase_scores, old, joblib, METHODS
from evaluate_shape import ROOT

POLICIES = ("original","veto","risk","phase_risk")
OUTPUTS = ("candidate_only","shared_edge")


def approval(score,q,cut,veto):
    if score is None or q is None:
        return dict.fromkeys(POLICIES,False)
    return dict(original=score>=cut["original"],
                veto=score>=cut["original"] and q<=veto,
                risk=score>=cut["risk"],phase_risk=score*(1-q)>=cut["phase_risk"])


def replay(track,source,model,frozen):
    rows, phases = source["rows"],phase_scores(source,model)
    receipt = old.read(PARENT/"result.json")
    path = PARENT/f"{track}-frames.jsonl"
    prior = receipt["tracks"][track]
    assert old.digest(path)==prior["rows_sha256"]
    originals = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    assert [(r["run"],r["frame"]) for r in originals]==source["keys"]
    phase_matrix = np.zeros((3,3),dtype=int)
    available = (source["phase_y"]>=0)&np.isfinite(phases).all(1)
    np.add.at(phase_matrix,(source["phase_y"][available],phases[available].argmax(1)),1)
    totals = {f"{method}_{policy}_{base}":defaultdict(list)
              for method in METHODS for policy in POLICIES for base in OUTPUTS}
    paired = {p:Counter() for p in POLICIES}
    details = []
    for i,(row,previous) in enumerate(zip(rows,originals,strict=True)):
        assert (row["truth"],row["label"],row["truth_endpoints_m"])==(
             previous["truth"],previous["label"],previous["truth_endpoints_m"])
        q = float(phases[i,2]) if np.isfinite(phases[i]).all() else None
        base_warning = previous["common_edge"]
        predictions, states = {},{}
        for method in METHODS:
            item = previous["scores"][method]
            score,candidate = item["score"],item["candidate"]
            assert score is None or q is not None
            accepted = approval(score,q,frozen["cuts"][method],frozen["veto_corner_probability_max"])
            for policy in POLICIES:
                offer = candidate if accepted[policy] else None
                for base,forecast in (("candidate_only",offer),("shared_edge",base_warning or offer)):
                    name = f"{method}_{policy}_{base}"
                    was = previous["states"][f"{method}_{base}"]
                    observed = forecast is not None or was=="straight"
                    published = old.publish(row,forecast,observed,"phase_or_candidate_uncertain")
                    totals[name][row["run"]].append(published)
                    predictions[name], states[name] = forecast,published["state"]
                    if policy=="original":
                        assert forecast==previous["final"][f"{method}_{base}"]
        for policy in POLICIES:
            a,b = (old.category(row,predictions[f"{method}_{policy}_candidate_only"])
                   for method in ("temporal","t4t5_confirm"))
            if a!=b:
                paired[policy][row["truth"],row["label"],a,b] += 1
        details.append(dict(run=row["run"],frame=row["frame"],truth=row["truth"],label=row["label"],
                            truth_endpoints_m=row["truth_endpoints_m"],common_edge=base_warning,
                            phase=None if q is None else phases[i].tolist(),scores=previous["scores"],
                            predictions=predictions,states=states))
    entry = old.context()
    runner = entry.import_replay()
    metrics = {m:runner.totals(data) for m,data in totals.items()}
    for m in METHODS:
        for mode in OUTPUTS:
            assert metrics[f"{m}_original_{mode}"]==prior["metrics"][f"{m}_{mode}"]
    output = OUT/f"{track}-frames.jsonl"
    old.jsonl(output,details)
    return dict(frames=len(details),phase_confusion=phase_matrix.tolist(),metrics=metrics,
                rows_sha256=old.digest(output),control_rows_sha256=old.digest(path),
                paired=[dict(policy=p,truth=k[0],label=k[1],edge=k[2],t4t5=k[3],frames=v)
                        for p,counts in paired.items() for k,v in sorted(counts.items())])


def main():
    freeze,checkpoint = OUT/"training.json",OUT/"phase.joblib"
    trained = old.read(freeze)
    assert trained["excluded"]==[SILVER,VAL] and trained["development"]==list(DEVELOP)
    assert old.digest(checkpoint)==trained["checkpoint_sha256"]
    for name,sha in trained["source_sha256"].items():
        assert SILVER not in name and VAL not in name and old.digest(ROOT/name)==sha,name
    model = joblib.load(checkpoint)
    sources = {Path(__file__),freeze,checkpoint,PARENT/"result.json",Path(phase_scores.__code__.co_filename)}
    reports = {}
    for track in (SILVER,VAL):
        source = group(track,sources)
        reports[track] = replay(track,source,model,trained)
        print(track,{k:v for k,v in reports[track]["metrics"].items()
                     if k.endswith("candidate_only") and not k.startswith("static")},flush=True)
    old.write(OUT/"result.json",dict(status="two_excluded_circuit_replays",seed=trained["seed"],
            source_revision=old.revision(),tracks=reports,cuts=trained["cuts"],
            phase_veto=trained["veto_corner_probability_max"],
            source_sha256={str(p):old.digest(p) for p in sorted(sources)},
            termination="phase_guard_and_same_candidates_verified"))


if __name__=="__main__":
    main()
