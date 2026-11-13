"""Read straight from visual geometry and local motion after turn confirmation."""
from pathlib import Path

import numpy as np

from routing import OUT, DEVELOP, SILVER, VAL, SEED, old, joblib
from directional_features import group
from verify_directional import frames
from sklearn.ensemble import HistGradientBoostingClassifier

ROUTE = "empty_or_consistent_calibrated"
METHODS = ("temporal", "t4t5_confirm")
PHASE_LABELS = ("approach", "in_corner")
STATES = ("straight", "left", "right", "abstain")


def observed(g):
    paired = np.isfinite(g["sections"]).all(axis=-1).sum(axis=1) >= 6
    return paired & np.isfinite(g["phase_x"]).all(axis=1)


def truth(g):
    return np.array([row["truth"] for row in g["rows"]])


def fit(groups):
    matrices, targets, weights = [], [], []
    for g in groups:
        labels = truth(g)
        valid = observed(g) & np.isin(labels, ("straight", *PHASE_LABELS))
        x, y = g["phase_x"][valid], (labels[valid] == "straight").astype(int)
        assert set(y) == {0, 1}
        matrices.append(x)
        targets.append(y)
        weights.append(1./np.bincount(y)[y])
    x, y, weight = (np.concatenate(a) for a in (matrices, targets, weights))
    weight *= len(weight) / weight.sum()
    return HistGradientBoostingClassifier(max_iter=75, max_leaf_nodes=12, min_samples_leaf=30,
             learning_rate=.08, l2_regularization=1., early_stopping=False,
             random_state=SEED).fit(x, y, sample_weight=weight)


def scores(g, model):
    valid = observed(g)
    result = np.full(len(valid), -1.)
    result[valid] = model.predict_proba(g["phase_x"][valid])[:, 1]
    return result


def calibrate(groups):
    probabilities, labels, before = [], [], []
    for held, g in groups.items():
        model = fit([v for k, v in groups.items() if k != held])
        probabilities.append(scores(g, model))
        labels.append(truth(g))
        before.append(observed(g) & np.array([not pool for pool in g["pools"]]))
    score, label, old_straight = (np.concatenate(a) for a in (probabilities, labels, before))
    masks = {name: label == name for name in ("straight", *PHASE_LABELS, "out_of_range")}
    cap = {name: int((masks[name] & old_straight).sum()) for name in PHASE_LABELS}
    options = np.r_[np.unique(score[score >= 0.]), np.nextafter(1., np.inf)]
    feasible = []
    for cut in options:
        fresh = score >= cut
        counts = {name: int((fresh & mask).sum()) for name, mask in masks.items()}
        if all(counts[name] <= cap[name] for name in PHASE_LABELS):
            feasible.append((counts["straight"], -sum(counts[name] for name in PHASE_LABELS), float(cut), counts))
    _, _, cut, report = max(feasible, key=lambda v: v[:3])
    return cut, dict(old={n: int((mask & old_straight).sum()) for n, mask in masks.items()},
                     new=report, separate_false_caps=cap)


def replay(rows, score, cut, methods=METHODS):
    new, summary = [], {m: {s: {} for s in ("old", "new")} for m in methods}
    for row, probability in zip(rows, score, strict=True):
        states = {}
        for method in methods:
            before = row["states"][method+"_candidate_only"]
            after = before if before in ("left", "right") else ("straight" if probability >= cut else "abstain")
            assert before in STATES and after in STATES
            assert (before == after) or row["scores"][method]["accepted"] is None
            for phase, state in (("old", before), ("new", after)):
                counts = summary[method][phase].setdefault(row["truth"]+"/"+str(row["label"]), dict.fromkeys(STATES, 0))
                counts[state] += 1
            states[method] = dict(old=before, new=after)
        new.append(dict(run=row["run"], frame=row["frame"], truth=row["truth"], label=row["label"],
                        straight_score=None if probability < 0. else float(probability), states=states))
    return new, summary


def main():
    sources = {Path(__file__), Path(group.__code__.co_filename), OUT/"training.json"}
    dev = {track: group(track, sources) for track in DEVELOP}
    cut, calibration = calibrate(dev)
    model = fit(list(dev.values()))
    checkpoint = OUT/"straight.joblib"
    joblib.dump(model, checkpoint)
    training = OUT/"straight-training.json"
    assert not any(SILVER in str(p) or VAL in str(p) for p in sources)
    old.write(training, dict(status="frozen_before_replay", seed=SEED, source_revision=old.revision(),
              development=DEVELOP, excluded=[SILVER, VAL], straight_cut=cut, calibration=calibration,
              route=ROUTE, checkpoint_sha256=old.digest(checkpoint),
              source_sha256={str(p): old.digest(p) for p in sorted(sources)}, termination="crossfit_then_freeze"))
    route_path = OUT/"result.json"
    result = old.read(route_path)
    reports = {}
    print("Frozen straight", cut, calibration, flush=True)
    for track in (SILVER, VAL):
        hashes = {training, checkpoint, route_path}
        g = group(track, hashes)
        path = OUT/f"{track}-{ROUTE}.jsonl"
        rows = frames(path, result["tracks"][track]["variants"][ROUTE]["rows_sha256"])
        assert [(r["run"], r["frame"]) for r in rows] == g["keys"]
        stream, summary = replay(rows, scores(g, model), cut)
        output = OUT/f"{track}-straight-frames.jsonl"
        old.jsonl(output, stream)
        hashes.add(path)
        reports[track] = dict(metrics=summary, rows_sha256=old.digest(output),
                              source_sha256={str(p): old.digest(p) for p in sorted(hashes)})
        print(track, {v: summary["t4t5_confirm"][v]["straight/0"] for v in ("old", "new")}, flush=True)
    old.write(OUT/"straight-result.json", dict(status="frozen_straight_replay", seed=SEED,
              training_sha256=old.digest(training), route_sha256=old.digest(route_path),
              tracks=reports, termination="straight_and_abstain_replayed_with_turns_preserved"))


if __name__ == "__main__":
    main()
