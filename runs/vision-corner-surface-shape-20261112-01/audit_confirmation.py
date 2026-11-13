"""Audit confirmation score ordering on Silverstone without choosing an operating threshold."""
from collections import Counter
from pathlib import Path
import json
import math

from shape_net import OUT
from prepare import SILVER, old

BASE_BUDGETS = (85, 117, 136)


def count_offer(row, offer):
    result = Counter()
    if offer is None:
        return result
    result[old.category(row,offer)] += 1
    if row["truth"]=="approach" and row["label"]=="2" and offer[0]==2:
        result["right_direction"] += 1
        if old.category(row,offer)=="joint_hit":
            result["right_joint"] += 1
    return result


def score_path(base, candidates):
    total = base.copy()
    points = [dict(threshold=None,counts=dict(total))]
    for value in sorted(candidates,reverse=True):
        total.update(candidates[value])
        points.append(dict(threshold=value,counts=dict(total)))
    return points


def outside(counts):
    return counts.get("straight_alert",0)+counts.get("out_of_range_alert",0)


def main():
    entry = old.context()
    runner = entry.import_replay()
    hashes = {Path(__file__),Path(runner.__file__),Path(old.signals.__code__.co_filename)}
    truth, priors, _, _, profiles, _ = runner.setup(SILVER,hashes)
    frames, receipt = OUT/"silverstone-frames.jsonl",OUT/"result.json"
    hashes.update((frames,receipt))
    result = old.read(receipt)
    assert old.digest(frames)==result["rows_sha256"]
    rows = [json.loads(line) for line in frames.read_text(encoding="utf8").splitlines()]
    measured, focus = {}, []
    for mode in ("shape_geometry","shape_heat","head","head_supported"):
        measured[mode] = {}
        for model,spec in runner.MODELS.items():
            baseline,candidates = Counter(),{}
            for row,prior,raw in zip(truth,priors,rows,strict=True):
                key = runner.key(row)
                assert key==runner.key(raw)
                base = prior["proposals"][spec[1]]
                baseline.update(count_offer(row,base))
                offer = raw["offers"].get(mode)
                if base or offer is None:
                    continue
                profile = profiles[model].get(key)
                value = old.signals(profile,*offer)["local"]
                if value is not None and math.isfinite(value):
                    candidates.setdefault(value,Counter()).update(count_offer(row,offer))
                if row["truth"]=="approach" and row["label"]=="2":
                    focus.append(dict(mode=mode,model=model,frame=key[1],truth_m=row["truth_endpoints_m"],
                                      candidate=offer,score=value,category=old.category(row,offer)))
            path = score_path(baseline,candidates)
            fixed = [p for p in path if p["threshold"] is None or p["threshold"]>=.25][-1]["counts"]
            control = result["metrics"][f"{model}_{mode}"]
            assert outside(fixed)==control["outside"] and fixed.get("joint_hit",0)==control["joint"]
            budgets = {}
            for budget in BASE_BUDGETS:
                eligible = [p["counts"] for p in path if outside(p["counts"])<=budget]
                if eligible:
                    best_right = max(c.get("right_joint",0) for c in eligible)
                    best_any = max(c.get("joint_hit",0) for c in eligible)
                    budgets[budget] = dict(max_right_joint=best_right,max_all_joint=best_any)
            measured[mode][model] = dict(fixed=fixed,within_observed_false_alert_budgets=budgets)
    report = dict(status="retrospective_score_ordering_only",seed=None,source_revision=old.revision(),
                  scores=measured,right_candidates=focus,
                  source_sha256={str(p):old.digest(p) for p in sorted(hashes)},
                  termination="fixed_score_reproduced_and_diagnostic_rank_measured")
    old.write(OUT/"confirmation-ordering.json",report)
    print(json.dumps(measured["shape_geometry"],indent=2),flush=True)


if __name__ == "__main__":
    main()
