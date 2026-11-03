"""Convert a replicated local proposal into a fresh-seed parallel main campaign."""

import argparse
import copy
import json
from pathlib import Path

from flybrain_drive.evidence import write_json


def prepare_main(proposal, policy):
    if policy.get("schema") != "flybrain/local-to-remote-main-v1":
        raise ValueError("Unknown promotion policy schema")
    selection = proposal["local_selection"]
    baseline = selection["baseline"]
    eligible = selection["remote_addition_candidates"]
    evidence = {row["arm"]: row for row in selection["evidence"]}
    cap = int(policy["max_candidates"])
    if cap < 1 or not eligible:
        raise ValueError("Main training requires replicated eligible candidates")
    ranks = []
    used_seeds = set()
    for name in eligible:
        row = evidence[name]
        paired = row["paired_seeds"]
        seeds = {pair["seed"] for pair in paired}
        if (not row["screen_passed"] or not row["replicated"] or len(seeds) < 2
                or not all(pair["passed"] for pair in paired)):
            raise ValueError("Candidate lacks passing independent paired replication")
        used_seeds.update(seeds)
        gain = min(pair["mean_progress_gain_m"] /
                   max(abs(pair["baseline_mean_progress_m"]), 1e-6) for pair in paired)
        ranks.append({"arm": name, "worst_seed_relative_gain": gain})
    ranks.sort(key=lambda row: (-row["worst_seed_relative_gain"], row["arm"]))
    chosen = [row["arm"] for row in ranks[:cap]]
    config = copy.deepcopy(proposal)
    by_name = {arm["name"]: arm for arm in config["campaign"]["arms"]}
    if len(by_name) != len(config["campaign"]["arms"]) or any(
        name not in by_name for name in [baseline, *chosen]
    ):
        raise ValueError("Main proposal is missing or duplicates selected arms")
    training = policy["training"]
    for name in ("num_envs", "horizon", "updates", "torch_threads"):
        value = int(training[name])
        if value < (2 if name == "num_envs" else 1):
            raise ValueError("Invalid main training budget")
        config[name] = value
    seeds = training["seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or used_seeds.intersection(seeds):
        raise ValueError("Main training needs at least three distinct fresh seeds")
    config["campaign"]["seeds"] = list(seeds)
    config["campaign"]["arms"] = [by_name[name] for name in [baseline, *chosen]]
    # Local dataset paths must never become dependencies of a remote job.
    for settings in [config.get("behavior_cloning", {}), *[
        arm.get("overrides", {}).get("behavior_cloning", {})
        for arm in config["campaign"]["arms"]
    ]]:
        settings.pop("precollected_teacher", None)
    config["purpose"] = "Replicated local shortlist; fresh-seed parallel remote main training"
    config["main_training"] = {
        "promotion_policy": copy.deepcopy(policy), "ranking": ranks, "selected": chosen,
        "development_seeds": sorted(used_seeds),
        "transitions_per_task": config["num_envs"] * config["horizon"] * config["updates"],
        "task_count": len(seeds) * (1 + len(chosen)),
        "initialization": "fresh seed; remote BC/DAgger then PPO; no local checkpoint warmstart",
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, prepare_main(json.loads(args.proposal.read_text(encoding="utf-8")),
                                         json.loads(args.policy.read_text(encoding="utf-8"))))
