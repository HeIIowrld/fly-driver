"""Calibrate aligned boundary alternatives before replaying two excluded circuits."""
from pathlib import Path

import numpy as np

from aligned_sections import (OUT, LISTWISE, expand, old, valid_pairs, MAX_AGE_S, MAX_HISTORY,
                              ANCHOR_RADIUS_M, ANCHOR_RESIDUAL_M, PAST_AGREEMENT_M)
from listwise import DEVELOP, SILVER, VAL, SEED, METHODS, train, predict, labels_for, risk_cut
from listwise import analyze, with_shared_veto, joblib


def added_stats(group, base, patches):
    selected = np.array([r["truth"] == "approach" and r["label"] == "2" for r in group["rows"]])
    added = [[offer for offer in full if offer not in earlier]
             for full, earlier in zip(group["pools"], base["pools"], strict=True)]
    before = np.array([any(old.category(row, o) == "joint_hit" for o in pool)
                       for row, pool in zip(group["rows"], base["pools"], strict=True)])
    after = np.array([any(old.category(row, o) == "joint_hit" for o in pool)
                      for row, pool in zip(group["rows"], group["pools"], strict=True)])
    assert np.all(after | ~before)
    ids = np.flatnonzero(after & ~before)
    return dict(frames_with_fills=int(patches.any(axis=1).sum()), filled_pairs=int(patches.sum()),
                right_frames_with_fills=int(patches[selected].any(axis=1).sum()),
                right_filled_pairs=int(patches[selected].sum()), added_candidates=sum(map(len, added)),
                new_joint_frames=int(len(ids)), new_right_joint_frames=int(selected[ids].sum()),
                recovered=[dict(run=group["keys"][i][0], frame=int(group["keys"][i][1]),
                                truth=group["rows"][i]["label"], truth_m=group["rows"][i]["truth_endpoints_m"],
                                candidates=[o for o in added[i] if old.category(group["rows"][i], o) == "joint_hit"])
                           for i in ids])


def main():
    sources = {Path(__file__), Path(expand.__code__.co_filename), Path(train.__code__.co_filename),
               Path(risk_cut.__code__.co_filename)}
    baseline_training = LISTWISE/"training.json"
    baseline = old.read(baseline_training)
    baseline_checkpoint = LISTWISE/"rankers.joblib"
    assert old.digest(baseline_checkpoint) == baseline["checkpoint_sha256"]
    assert baseline["development"] == list(DEVELOP) and baseline["excluded"] == [SILVER, VAL]
    caps, limit = baseline["development_caps"], baseline["shared_phase_veto"]
    groups, stats = {}, {}
    for track in DEVELOP:
        group, base, patches = expand(track, sources)
        groups[track], stats[track] = group, added_stats(group, base, patches)
        print("Development", track, stats[track], flush=True)
    folds = {track: predict(group, train([g for k, g in groups.items() if k != track]))
             for track, group in groups.items()}
    cuts = {}
    for method in METHODS:
        scores = np.array([score for track in DEVELOP for _, _, score in folds[track][method]])
        labels = np.concatenate([labels_for(groups[track], folds[track][method]) for track in DEVELOP])
        cuts[method] = risk_cut(scores, labels, caps["outside"], caps["in_corner"])
    models = train(list(groups.values()))
    checkpoint = OUT/"rankers.joblib"
    joblib.dump(models, checkpoint)
    training = OUT/"training.json"
    sources.add(baseline_training)
    assert not any(SILVER in str(p) or VAL in str(p) for p in sources)
    old.write(training, dict(status="frozen_before_replay", development=DEVELOP, excluded=[SILVER, VAL],
              seed=SEED, source_revision=old.revision(), development_caps=caps, shared_phase_veto=limit,
              cuts=cuts, development_stats=stats, history_s=MAX_AGE_S, history_frames=MAX_HISTORY,
              anchor_radius_m=ANCHOR_RADIUS_M, anchor_residual_m=ANCHOR_RESIDUAL_M,
              past_agreement_m=PAST_AGREEMENT_M, checkpoint_sha256=old.digest(checkpoint),
              source_sha256={str(p): old.digest(p) for p in sorted(sources)}, termination="crossfit_then_freeze"))
    print("Frozen", cuts, flush=True)
    prior_models = joblib.load(baseline_checkpoint)
    baseline_result = old.read(LISTWISE/"result.json")
    reports = {}
    for track in (SILVER, VAL):
        hashes = {training, checkpoint, baseline_checkpoint, LISTWISE/"result.json",
                  Path(analyze.__code__.co_filename)}
        group, base, patches = expand(track, hashes)
        variants = (("direct", base, prior_models, baseline["cuts"]),
                    ("carry_frozen", group, prior_models, baseline["cuts"]),
                    ("carry_refit", group, models, cuts))
        report = dict(added=added_stats(group, base, patches), variants={})
        sections = OUT/f"{track}-sections.npz"
        np.savez_compressed(sections, run=[key[0] for key in group["keys"]],
                            frame=[key[1] for key in group["keys"]], stations=old.STATIONS,
                            current=base["sections"], aligned=group["sections"], added=patches)
        report["sections_sha256"] = old.digest(sections)
        for name, source, model, threshold in variants:
            picks = predict(source, model)
            evaluated, rows = analyze(source, with_shared_veto(track, source, picks, limit, hashes), threshold)
            output = OUT/f"{track}-{name}.jsonl"
            old.jsonl(output, rows)
            evaluated["rows_sha256"] = old.digest(output)
            if name == "direct":
                assert evaluated["metrics"] == baseline_result["tracks"][track]["veto_metrics"]
                assert evaluated["rows_sha256"] == baseline_result["tracks"][track]["veto_rows_sha256"]
            assert len(rows) == 1200
            report["variants"][name] = evaluated
        report["source_sha256"] = {str(p): old.digest(p) for p in sorted(hashes)}
        reports[track] = report
        print(track, report["added"], {name: data["metrics"]["t4t5_confirm_candidate_only"]
              for name, data in report["variants"].items()}, flush=True)
    old.write(OUT/"result.json", dict(status="two_frozen_replays", seed=SEED, source_revision=old.revision(),
              training_sha256=old.digest(training), tracks=reports, termination="boundary_carry_replays_scored"))


if __name__ == "__main__":
    main()
