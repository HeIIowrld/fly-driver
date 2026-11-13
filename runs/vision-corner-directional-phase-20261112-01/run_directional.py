"""Fit signed candidate context on development tracks, then freeze and replay."""
from collections import Counter
from pathlib import Path
import json

import numpy as np

from directional_features import OUT, PHASE, DEVELOP, SILVER, VAL, enrich, old
from train_phase import risk_cut, SEED
from rank import MODELS, METHODS, fit, choose, labels_for, joblib
from evaluate_rank import analyze

ROOT = OUT.parents[1]


def train(sources,caps):
    groups = {}
    for track in DEVELOP:
        g = enrich(track,sources)
        groups[track] = g
        print(track,len(g["index"]),int(g["target"].sum()),flush=True)
    folds = {}
    for track in DEVELOP:
        others = [g for name,g in groups.items() if name!=track]
        folds[track] = choose(groups[track],{mode:fit(others,mode) for mode in MODELS})
    cuts,calibration = {},{}
    for method in METHODS:
        probability = np.asarray([p for track in DEVELOP for _,_,p in folds[track][method]])
        labels = np.concatenate([labels_for(groups[track],folds[track][method]) for track in DEVELOP])
        cuts[method] = risk_cut(probability,labels,*caps)
        calibration[method] = dict(Counter(labels[probability>=cuts[method]].tolist()))
    return {mode:fit(list(groups.values()),mode) for mode in MODELS},cuts,calibration


def with_shared_veto(track,g,picks,limit,hashes):
    path = PHASE/f"{track}-frames.jsonl"
    receipt = PHASE/"result.json"
    assert old.digest(path)==old.read(receipt)["tracks"][track]["rows_sha256"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    assert [(r["run"],r["frame"]) for r in rows]==g["keys"]
    hashes.update((path,receipt))
    return {m:[(frame,j,p if rows[frame]["phase"][2]<=limit else -1.)
                for frame,j,p in seq] for m,seq in picks.items()}


def main():
    yaw = OUT/"yaw-sign.json"
    sign = old.read(yaw)
    assert set(sign["tracks"])==set(DEVELOP)
    assert all(x["positive_left_agreement"]>=.85 for x in sign["tracks"].values())
    for name,sha in sign["source_sha256"].items():
        assert SILVER not in name and VAL not in name and old.digest(ROOT/name)==sha
    guard = PHASE/"training.json"
    frozen_guard = old.read(guard)
    caps = tuple(frozen_guard["development_false_caps"][k] for k in ("outside","in_corner"))
    limit = frozen_guard["veto_corner_probability_max"]
    sources = {Path(__file__),yaw,guard,Path(enrich.__code__.co_filename),Path(fit.__code__.co_filename),
               Path(choose.__code__.co_filename),Path(risk_cut.__code__.co_filename)}
    weights,cuts,calibration = train(sources,caps)
    checkpoint = OUT/"rankers.joblib"
    joblib.dump(weights,checkpoint)
    manifest = OUT/"training.json"
    assert not any(SILVER in str(p) or VAL in str(p) for p in sources)
    old.write(manifest,dict(status="frozen_before_test",seed=SEED,source_revision=old.revision(),
             development=DEVELOP,excluded=[SILVER,VAL],cuts=cuts,development_caps=dict(outside=caps[0],in_corner=caps[1]),
             calibration=calibration,shared_phase_veto=limit,checkpoint_sha256=old.digest(checkpoint),
             source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="crossfit_then_freeze"))
    print("Development",calibration,flush=True)
    reports = {}
    for track in (SILVER,VAL):
        hashes = {Path(__file__),manifest,checkpoint,Path(enrich.__code__.co_filename),Path(analyze.__code__.co_filename)}
        g = enrich(track,hashes)
        picks = choose(g,weights)
        report,records = analyze(g,picks,cuts)
        guarded,g_records = analyze(g,with_shared_veto(track,g,picks,limit,hashes),cuts)
        assert len(records)==len(g_records)==1200
        path,vpath = OUT/f"{track}-frames.jsonl",OUT/f"{track}-veto-frames.jsonl"
        old.jsonl(path,records)
        old.jsonl(vpath,g_records)
        report.update(rows_sha256=old.digest(path),veto_rows_sha256=old.digest(vpath),
                      veto_metrics=guarded["metrics"],veto_paired=guarded["paired"],
                      source_sha256={str(p):old.digest(p) for p in sorted(hashes)})
        reports[track] = report
        print(track,{k:v for k,v in report["metrics"].items() if k.endswith("candidate_only")},flush=True)
    old.write(OUT/"result.json",dict(status="two_frozen_replays",seed=SEED,source_revision=old.revision(),
              cuts=cuts,tracks=reports,training_sha256=old.digest(manifest),
              termination="same_candidates_signed_context_and_veto_scored"))


if __name__=="__main__":
    main()
