"""Fit a grouped onset selector with separate absolute confirmation."""
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent/"vision-corner-directional-phase-20261112-01"
sys.path.insert(0, str(PREVIOUS))
from run_directional import DEVELOP, SILVER, VAL, PHASE, SEED, enrich, risk_cut, analyze, with_shared_veto, old
from rank import MODELS, METHODS, VALID_CATEGORIES, fit, encode, labels_for, joblib

PENALTY = .1


def groups_for_ranking(group):
    known = np.isin(group["errors"], VALID_CATEGORIES)
    indices = np.flatnonzero(known)
    frames = group["index"][indices]
    breaks = np.flatnonzero(np.diff(frames)) + 1
    for segment in np.split(indices, breaks):
        good = group["target"][segment]
        if good.any() and not good.all():
            yield segment, good


def objective(weights, frames):
    loss = .5 * PENALTY * (weights @ weights)
    gradient = PENALTY * weights
    for x, target, share in frames:
        scores = x @ weights
        norm_all = logsumexp(scores)
        norm_good = logsumexp(scores[target])
        loss += share * (norm_all - norm_good)
        gradient += share * (np.exp(scores - norm_all) @ x -
                             np.exp(scores[target] - norm_good) @ x[target])
    return loss, gradient


def train_selector(groups, mode, confirmer):
    examples = []
    pair_counts = dict(same_direction=0, opposite_direction=0)
    for group in groups:
        encoded = encode(group, confirmer, mode)
        for segment, good in groups_for_ranking(group):
            direction = group["offers"][segment[np.flatnonzero(good)[0]]][0]
            assert all(group["offers"][i][0] == direction for i in segment[good])
            for i in segment[~good]:
                key = "same_direction" if group["offers"][i][0] == direction else "opposite_direction"
                pair_counts[key] += 1
            examples.append((encoded[segment], good, direction))
    count_by_direction = {d: sum(e[2] == d for e in examples) for d in (1, 2)}
    assert all(count_by_direction.values())
    frames = [(x, good, .5 / count_by_direction[d]) for x, good, d in examples]
    start = np.zeros(frames[0][0].shape[1])
    solved = minimize(objective, start, args=(frames,), method="L-BFGS-B", jac=True,
                      options={"maxiter": 400, "ftol": 1.e-10, "gtol": 1.e-6})
    assert solved.success and np.isfinite(solved.x).all(), solved.message
    return dict(weights=solved.x, frames=count_by_direction, negatives=pair_counts,
                loss=float(solved.fun), iterations=int(solved.nit))


def train(groups):
    models = {}
    for mode in MODELS:
        confirmer = fit(groups, mode)
        models[mode] = dict(confirmer=confirmer, selector=train_selector(groups, mode, confirmer))
    return models


def predict(group, models):
    rank, confirm = {}, {}
    for mode, model in models.items():
        features = encode(group, model["confirmer"], mode)
        rank[mode] = features @ model["selector"]["weights"]
        confirm[mode] = model["confirmer"]["model"].predict_proba(features)[:, 1]
    output = {method: [] for method in METHODS}
    indices = group["index"]
    for segment in np.split(np.arange(len(indices)), np.flatnonzero(np.diff(indices)) + 1):
        if not len(segment):
            continue
        chosen = {mode: int(segment[np.argmax(rank[mode][segment])]) for mode in MODELS}
        for method, selector, confirmer in (("static", "static", "static"),
                ("temporal", "temporal", "temporal"), ("t4t5_confirm", "temporal", "t4t5"),
                ("t4t5_rerank", "t4t5", "t4t5")):
            i = chosen[selector]
            output[method].append((int(indices[i]), i, float(confirm[confirmer][i])))
    return output


def main():
    assert not set(DEVELOP) & {SILVER, VAL}
    sources = {Path(__file__), Path(enrich.__code__.co_filename), Path(fit.__code__.co_filename),
               Path(risk_cut.__code__.co_filename)}
    prior_path = PREVIOUS/"training.json"
    sources.add(prior_path)
    prior = old.read(prior_path)
    caps = prior["development_caps"]
    limit = prior["shared_phase_veto"]
    groups = {track: enrich(track, sources) for track in DEVELOP}
    frozen = {}
    for track, current in groups.items():
        frozen[track] = predict(current, train([g for key, g in groups.items() if key != track]))
        print("Crossfit", track, flush=True)
    cuts = {}
    for method in METHODS:
        scores = np.array([score for t in DEVELOP for _, _, score in frozen[t][method]])
        labels = np.concatenate([labels_for(groups[t], frozen[t][method]) for t in DEVELOP])
        cuts[method] = risk_cut(scores, labels, caps["outside"], caps["in_corner"])
    models = train(list(groups.values()))
    checkpoint = OUT/"rankers.joblib"
    joblib.dump(models, checkpoint)
    training = OUT/"training.json"
    assert not any(SILVER in str(p) or VAL in str(p) for p in sources)
    old.write(training, dict(status="frozen_before_replay", seed=SEED, source_revision=old.revision(),
              development=DEVELOP, excluded=[SILVER, VAL], cuts=cuts, development_caps=caps,
              selector_penalty=PENALTY, shared_phase_veto=limit,
              selector={mode: {k: v for k, v in model["selector"].items() if k != "weights"}
                        for mode, model in models.items()},
              checkpoint_sha256=old.digest(checkpoint), source_sha256={str(p): old.digest(p) for p in sorted(sources)},
              termination="crossfit_then_freeze"))
    print("Frozen", cuts, flush=True)
    reports = {}
    for track in (SILVER, VAL):
        hashes = {training, checkpoint, Path(analyze.__code__.co_filename)}
        group = enrich(track, hashes)
        picks = predict(group, models)
        report, rows = analyze(group, picks, cuts)
        guarded, veto = analyze(group, with_shared_veto(track, group, picks, limit, hashes), cuts)
        assert len(rows) == len(veto) == 1200
        path, veto_path = OUT/f"{track}-frames.jsonl", OUT/f"{track}-veto-frames.jsonl"
        old.jsonl(path, rows)
        old.jsonl(veto_path, veto)
        report.update(rows_sha256=old.digest(path), veto_rows_sha256=old.digest(veto_path),
                      veto_metrics=guarded["metrics"], veto_paired=guarded["paired"],
                      source_sha256={str(p): old.digest(p) for p in sorted(hashes)})
        reports[track] = report
        print(track, report["pre_cut_right"], {k: v for k, v in guarded["metrics"].items()
              if k.endswith("candidate_only")}, flush=True)
    old.write(OUT/"result.json", dict(status="two_frozen_replays", seed=SEED, source_revision=old.revision(),
              cuts=cuts, tracks=reports, training_sha256=old.digest(training),
              termination="listwise_selection_and_absolute_confirmation_scored"))


if __name__ == "__main__":
    main()
