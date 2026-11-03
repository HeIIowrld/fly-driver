"""Predeclared paired development-screen gate; never replace remote baseline trials."""

import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import write_json


def select_candidates(summary, baseline="full_mixed"):
    rows = summary["runs"]
    completed = {(r["arm"], r["seed"]): r for r in rows if r["status"] == "completed"}
    names = sorted({r["arm"] for r in rows} - {baseline})
    seeds = sorted({r["seed"] for r in rows})
    evidence = []
    for name in names:
        paired = []
        for seed in seeds:
            a, b = completed.get((baseline, seed)), completed.get((name, seed))
            if a is None or b is None:
                paired.append({"seed": seed, "passed": False, "reason": "missing_or_failed_run"})
                continue
            ac, bc = a["final"]["cases"], b["final"]["cases"]
            if [c["case"] for c in ac] != [c["case"] for c in bc]:
                raise ValueError("Paired development cases do not match")
            ag, bg = a["final"]["aggregate"], b["final"]["aggregate"]
            improved = sum(cb["progress_delta"] > ca["progress_delta"]
                           for ca, cb in zip(ac, bc, strict=True))
            af = sum(c["failure"] for c in ac)
            bf = sum(c["failure"] for c in bc)
            gain = bg["mean_progress_delta"] - ag["mean_progress_delta"]
            passed = (bf <= af and bg["mean_progress_delta"] >=
                      max(1.25 * ag["mean_progress_delta"], ag["mean_progress_delta"] + 1e-6)
                      and improved >= 3)
            paired.append({"seed": seed, "passed": passed, "improved_cases": improved,
                           "baseline_failures": af, "candidate_failures": bf,
                           "baseline_mean_progress_m": ag["mean_progress_delta"],
                           "candidate_mean_progress_m": bg["mean_progress_delta"],
                           "mean_progress_gain_m": gain})
        evidence.append({"arm": name, "paired_seeds": paired,
                         "screen_passed": bool(paired) and all(p["passed"] for p in paired),
                         "replicated": len(seeds) >= 2})
    return {
        "baseline": baseline,
        "selection_cases": "development validation, separate from remote test suite",
        "rule": "no extra failures, >=25% mean progress gain, >=3/4 improved paired cases",
        "evidence": evidence,
        "replication_candidates": [e["arm"] for e in evidence if e["screen_passed"]],
        "remote_addition_candidates": [e["arm"] for e in evidence
                                       if e["screen_passed"] and e["replicated"]],
        "remote_existing_campaign": "unchanged; additions require replicated pilot evidence",
        "interpretation": "priority screen only; does not establish autonomous driving success",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, select_candidates(json.loads(args.summary.read_text())))
