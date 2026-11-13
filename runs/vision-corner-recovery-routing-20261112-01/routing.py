"""Replay candidate routing policies with common edge and motion selection."""
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ALIGNED = OUT.parent/"vision-corner-aligned-sections-20261112-01"
sys.path.insert(0, str(ALIGNED))
from aligned_sections import LISTWISE, expand, old
from listwise import DEVELOP, SILVER, VAL, SEED, METHODS, train, predict, labels_for, risk_cut
from listwise import analyze, with_shared_veto, joblib
from alternatives import MAX_ALONG

POLICIES = ("empty_pool", "empty_or_consistent")
BASELINE_MODES = ("direct", "union")


def routed(group, base, models):
    original = predict(base, models)
    full = predict(group, models)
    union_index = {(int(i), candidate): j for j, (i, candidate) in
                   enumerate(zip(group["index"], group["offers"], strict=True))}
    direct_indexed = {name: [(i, union_index[i, base["offers"][j]], score) for i, j, score in picks]
                      for name, picks in original.items()}
    routes = {name: {} for name in (*BASELINE_MODES, *POLICIES)}
    routes["direct"] = direct_indexed
    routes["union"] = full
    for method in METHODS:
        direct = {p[0]: p for p in direct_indexed[method]}
        result = {policy: [] for policy in POLICIES}
        for alternative in full[method]:
            i, j, _ = alternative
            current = direct.get(i)
            empty = not base["pools"][i]
            if current is None:
                assert empty, (i, method)
                consistent = False
            else:
                expected, proposed = (group["offers"][k] for k in (current[1], j))
                consistent = expected[0] == proposed[0] and abs(expected[1]-proposed[1]) <= MAX_ALONG
            for policy, allowed in (("empty_pool", empty), ("empty_or_consistent", empty or consistent)):
                chosen = alternative if allowed else current
                if chosen is not None:
                    result[policy].append(chosen)
        for policy, options in result.items():
            routes[policy][method] = options
    return routes


def calibrate(groups, caps):
    folds = {}
    for held, (expanded, base) in groups.items():
        model = train([b for name, (_, b) in groups.items() if name != held])
        folds[held] = routed(expanded, base, model)
        print("Crossfit", held, flush=True)
    cuts, stats = {}, {}
    for policy in POLICIES:
        cuts[policy], stats[policy] = {}, {}
        for method in METHODS:
            picks = [folds[track][policy][method] for track in DEVELOP]
            scores = np.array([p for seq in picks for _, _, p in seq])
            labels = np.concatenate([labels_for(groups[t][0], seq) for t, seq in zip(DEVELOP, picks, strict=True)])
            cut = risk_cut(scores, labels, caps["outside"], caps["in_corner"])
            cuts[policy][method] = cut
            values, count = np.unique(labels[scores >= cut], return_counts=True)
            stats[policy][method] = dict(zip(values.tolist(), count.tolist(), strict=True))
    return cuts, stats


def main():
    base_training, base_model = LISTWISE/"training.json", LISTWISE/"rankers.joblib"
    prior = old.read(base_training)
    assert prior["checkpoint_sha256"] == old.digest(base_model)
    assert prior["development"] == list(DEVELOP) and prior["excluded"] == [SILVER, VAL]
    sources = {Path(__file__), Path(expand.__code__.co_filename), Path(train.__code__.co_filename),
               Path(risk_cut.__code__.co_filename), base_training, base_model}
    groups = {}
    for track in DEVELOP:
        expanded, base, _ = expand(track, sources)
        groups[track] = (expanded, base)
    cuts, stats = calibrate(groups, prior["development_caps"])
    trained = OUT/"training.json"
    assert not any(SILVER in str(p) or VAL in str(p) for p in sources)
    old.write(trained, dict(status="frozen_before_replay", seed=SEED, source_revision=old.revision(),
              development=DEVELOP, excluded=[SILVER, VAL], cuts=cuts, calibration=stats,
              baseline_cuts=prior["cuts"], development_caps=prior["development_caps"],
              max_agreement_distance_m=MAX_ALONG, shared_phase_veto=prior["shared_phase_veto"],
              source_sha256={str(p): old.digest(p) for p in sorted(sources)}, termination="routes_crossfit_then_frozen"))
    print("Frozen", cuts, flush=True)
    models = joblib.load(base_model)
    aligned_result = old.read(ALIGNED/"result.json")
    reports = {}
    for track in (SILVER, VAL):
        hashes = {trained, ALIGNED/"result.json", Path(analyze.__code__.co_filename)}
        expanded, base, _ = expand(track, hashes)
        picks = routed(expanded, base, models)
        report = {}
        for name, route, active, settings in [
                ("direct", "direct", base, prior["cuts"]),
                ("union", "union", expanded, prior["cuts"]),
                *((p+"_fixed", p, expanded, prior["cuts"]) for p in POLICIES),
                *((p+"_calibrated", p, expanded, cuts[p]) for p in POLICIES)]:
            options = predict(base, models) if name == "direct" else picks[route]
            output, rows = analyze(active, with_shared_veto(track, active, options, prior["shared_phase_veto"], hashes),
                                   settings)
            path = OUT/f"{track}-{name}.jsonl"
            old.jsonl(path, rows)
            output["rows_sha256"] = old.digest(path)
            if name in BASELINE_MODES:
                old_name = {"direct": "direct", "union": "carry_frozen"}[name]
                assert output["rows_sha256"] == aligned_result["tracks"][track]["variants"][old_name]["rows_sha256"]
            assert len(rows) == 1200
            report[name] = output
        reports[track] = dict(variants=report, source_sha256={str(p): old.digest(p) for p in sorted(hashes)})
        print(track, {name: r["metrics"]["t4t5_confirm_candidate_only"] for name, r in report.items()}, flush=True)
    old.write(OUT/"result.json", dict(status="two_frozen_replays", seed=SEED, source_revision=old.revision(),
              training_sha256=old.digest(trained), tracks=reports, termination="candidate_routing_replays_scored"))


if __name__ == "__main__":
    main()
