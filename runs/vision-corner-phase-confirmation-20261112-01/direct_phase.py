"""Compare learning directly from phase cues with a separate phase veto."""
from collections import Counter
from pathlib import Path
import numpy as np

from train_phase import OUT, DEVELOP, SILVER, VAL, group, old, risk_cut, SEED
from rank import MODELS, METHODS, fit, choose, labels_for, joblib
from evaluate_rank import analyze

DEST = OUT/"direct-phase"


def enriched(g):
    frame_features = g["phase_x"][g["index"]]
    assert np.isfinite(frame_features).all()
    return dict(g,edge=np.c_[g["edge"],frame_features])


def train(sources,caps):
    groups = {t:enriched(group(t,sources)) for t in DEVELOP}
    folds = {}
    for t in DEVELOP:
        others = [g for name,g in groups.items() if name!=t]
        folds[t] = choose(groups[t],{m:fit(others,m) for m in MODELS})
    cuts, calibration = {},{}
    for method in METHODS:
        scores = np.asarray([p for t in DEVELOP for _,_,p in folds[t][method]])
        labels = np.concatenate([labels_for(groups[t],folds[t][method]) for t in DEVELOP])
        cuts[method] = risk_cut(scores,labels,*caps)
        calibration[method] = dict(Counter(labels[scores>=cuts[method]].tolist()))
    return {m:fit(list(groups.values()),m) for m in MODELS},cuts,calibration


def main():
    DEST.mkdir(exist_ok=True)
    guard = OUT/"training.json"
    frozen = old.read(guard)
    caps = tuple(frozen["development_false_caps"][k] for k in ("outside","in_corner"))
    sources = {Path(__file__),guard,Path(group.__code__.co_filename),Path(risk_cut.__code__.co_filename),
               Path(fit.__code__.co_filename),Path(choose.__code__.co_filename)}
    weights,cuts,development = train(sources,caps)
    checkpoint = DEST/"rankers.joblib"
    joblib.dump(weights,checkpoint)
    manifest = DEST/"training.json"
    assert not any(SILVER in str(p) or VAL in str(p) for p in sources)
    old.write(manifest,dict(status="frozen_before_replays",seed=SEED,source_revision=old.revision(),
             development=DEVELOP,excluded=[SILVER,VAL],cuts=cuts,caps=dict(outside=caps[0],in_corner=caps[1]),
             calibration=development,checkpoint_sha256=old.digest(checkpoint),
             source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="crossfit_then_freeze"))
    print("Development",development,flush=True)
    outputs = {}
    for track in (SILVER,VAL):
        hashes = {Path(__file__),manifest,checkpoint,Path(group.__code__.co_filename),Path(analyze.__code__.co_filename)}
        g = enriched(group(track,hashes))
        report,records = analyze(g,choose(g,weights),cuts)
        assert len(records)==1200
        output = DEST/f"{track}-frames.jsonl"
        old.jsonl(output,records)
        report.update(rows_sha256=old.digest(output),
                      source_sha256={str(p):old.digest(p) for p in sorted(hashes)})
        outputs[track] = report
        print(track,{k:v for k,v in report["metrics"].items() if k.endswith("candidate_only")},flush=True)
    old.write(DEST/"result.json",dict(status="two_excluded_replays",seed=SEED,source_revision=old.revision(),
              cuts=cuts,tracks=outputs,training_sha256=old.digest(manifest),
              termination="direct_phase_features_scored_with_same_candidate_control"))


if __name__=="__main__":
    main()
