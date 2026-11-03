"""Bounded paired-seed follow-up: train, evaluate initial/final, record comparisons."""

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.training_validation import ensure_test_separation


def merged(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def paired_comparisons(rows, names):
    """Same-seed/same-case contrasts; retain individual deltas, make no significance claim."""
    completed = {(row["arm"], row["seed"]): row for row in rows if row["status"] == "completed"}
    pairs = []
    for left, right in zip(names[:-1], names[1:], strict=True):
        for seed in sorted({row["seed"] for row in rows}):
            if (left, seed) not in completed or (right, seed) not in completed:
                continue
            a = completed[left, seed]["final"]["cases"]
            b = completed[right, seed]["final"]["cases"]
            if [case["case"] for case in a] != [case["case"] for case in b]:
                raise ValueError("Comparison case order mismatch")
            for ca, cb in zip(a, b, strict=True):
                pairs.append(
                    {
                        "contrast": f"{right} minus {left}",
                        "seed": seed,
                        "case": ca["case"],
                        "progress_delta_difference": cb["progress_delta"] - ca["progress_delta"],
                        "completion_difference": int(cb["completion"]) - int(ca["completion"]),
                        "failure_difference": int(cb["failure"]) - int(ca["failure"]),
                        "lateral_rms_difference": cb["lateral_error_rms"] - ca["lateral_error_rms"],
                    }
                )
    return pairs


def campaign_for_index(config, index):
    """One independent seed/arm; array task order is seed-major, arm-minor."""
    result = copy.deepcopy(config)
    campaign = result["campaign"]
    arms, seeds = campaign["arms"], campaign["seeds"]
    count = len(arms) * len(seeds)
    if not 0 <= index < count:
        raise ValueError("Campaign array index is out of range")
    campaign["seeds"] = [seeds[index // len(arms)]]
    campaign["arms"] = [arms[index % len(arms)]]
    result["array_shard"] = index
    return result


def merge_shards(config, shards, output):
    """Retain missing/failed tasks explicitly; runs only after the full array terminates."""
    campaign = config["campaign"]
    count = len(campaign["arms"]) * len(campaign["seeds"])
    rows, hashes = [], {}
    for index in range(count):
        subset = campaign_for_index(config, index)["campaign"]
        seed, name = subset["seeds"][0], subset["arms"][0]["name"]
        summary = Path(shards) / f"task-{index}" / "campaign/summary.json"
        if summary.exists():
            hashes[str(summary)] = sha256(summary)
            values = json.loads(summary.read_text())["runs"]
            if len(values) != 1 or (values[0]["arm"], values[0]["seed"]) != (name, seed):
                raise ValueError("Shard identity does not match the sealed campaign")
            row = values[0]
        else:
            row = {
                "arm": name,
                "seed": seed,
                "status": "failed",
                "reason": "missing_shard_summary",
                "array_shard": index,
            }
        rows.append(row)
    run = begin_run(output, {"campaign_config": config, "shard_summary_sha256": hashes})
    write_json(
        run / "summary.json",
        {
            "runs": rows,
            "paired_comparisons": paired_comparisons(rows, [a["name"] for a in campaign["arms"]]),
            "interpretation": "paired pilot; missing tasks are failures, no superiority claim",
        },
    )
    failed = sum(row["status"] != "completed" for row in rows)
    write_json(
        run / "result.json",
        {
            "status": "completed" if not failed else "completed_with_failures",
            "runs": count,
            "failed": failed,
            "training_interactions": sum(
                row["training"]["interactions"] for row in rows if row["status"] == "completed"
            ),
        },
    )
    return failed


def run_campaign(config, output):
    # Spawn imports the entrypoint in every worker. Learner libraries belong only
    # in the parent, so CPU simulators do not each load PyTorch/ROCm.
    from flybrain_drive.evaluation import evaluate_suite
    from flybrain_drive.parallel_learning import train_parallel

    campaign = config["campaign"]
    arms, seeds = campaign["arms"], campaign["seeds"]
    names = [arm["name"] for arm in arms]
    if not names or len(names) != len(set(names)) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Campaign requires unique arms and seeds")
    if any(not re.fullmatch(r"[a-z0-9_-]+", name) for name in names):
        raise ValueError("Unsafe arm name")
    base = {key: value for key, value in config.items() if key != "campaign"}
    run = begin_run(output, config)
    rows = []
    for seed in seeds:
        for arm in arms:
            name = arm["name"]
            arm_output = run / name / f"seed-{int(seed)}"
            arm_output.mkdir(parents=True)
            arm_config = merged(base, arm.get("overrides", {}))
            arm_config["seed"] = int(seed)
            ensure_test_separation(arm_config.get("validation"), campaign["evaluation"])
            row = {"arm": name, "seed": int(seed), "status": "running"}
            write_json(run / "progress.json", {"current": row, "completed": rows})
            try:
                training = train_parallel(arm_config, arm_output / "training")
                initial = evaluate_suite(
                    training / "initial-checkpoint.pt",
                    arm_output / "initial-evaluation",
                    campaign["evaluation"],
                    record=bool(campaign.get("record_evaluation", False)),
                    record_every=int(campaign.get("record_every", 4)),
                )
                final = evaluate_suite(
                    training / "checkpoint.pt",
                    arm_output / "final-evaluation",
                    campaign["evaluation"],
                    record=bool(campaign.get("record_evaluation", False)),
                    record_every=int(campaign.get("record_every", 4)),
                )
                selected = None
                if (training / "best-checkpoint.pt").exists():
                    if sha256(training / "best-checkpoint.pt") == sha256(
                        training / "initial-checkpoint.pt"
                    ):
                        selected = initial
                        row["selected_evaluation_reused"] = "identical_initial_checkpoint"
                    else:
                        selected = evaluate_suite(
                            training / "best-checkpoint.pt", arm_output / "selected-evaluation",
                            campaign["evaluation"],
                            record=bool(campaign.get("record_evaluation", False)),
                            record_every=int(campaign.get("record_every", 4)),
                        )
                row.update(
                    status="completed",
                    initial=initial,
                    final=final,
                    training=json.loads((training / "training.json").read_text()),
                    mean_progress_gain=(
                        final["aggregate"]["mean_progress_delta"]
                        - initial["aggregate"]["mean_progress_delta"]
                    ),
                )
                if selected is not None:
                    row.update(
                        selected=selected,
                        checkpoint_selection=json.loads(
                            (training / "validation.json").read_text(encoding="utf-8")
                        ),
                        selected_mean_progress_gain=(
                            selected["aggregate"]["mean_progress_delta"]
                            - initial["aggregate"]["mean_progress_delta"]
                        ),
                    )
            except Exception as error:
                row.update(status="failed", reason=type(error).__name__, message=str(error))
            rows.append(row)
            write_json(run / "progress.json", {"current": None, "completed": rows})
            write_json(
                run / "summary.json",
                {
                    "runs": rows,
                    "paired_comparisons": paired_comparisons(rows, names),
                    "interpretation": "paired pilot; raw contrasts, no significance claim",
                },
            )
    failed = sum(row["status"] != "completed" for row in rows)
    completed_interactions = sum(
        row["training"]["interactions"] for row in rows if row["status"] == "completed"
    )
    write_json(
        run / "result.json",
        {
            "status": "completed" if not failed else "completed_with_failures",
            "runs": len(rows),
            "failed": failed,
            "training_interactions": completed_interactions,
            "mean_progress_gain": float(
                np.mean([row["mean_progress_gain"] for row in rows if row["status"] == "completed"])
            )
            if failed != len(rows)
            else None,
        },
    )
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, help="run one seed/arm as a Slurm array task")
    parser.add_argument(
        "--merge-shards", type=Path, help="merge task-N directories after termination"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.merge_shards is not None:
        if args.index is not None:
            parser.error("--index and --merge-shards are mutually exclusive")
        failed = merge_shards(config, args.merge_shards, args.output)
    else:
        if args.index is not None:
            config = campaign_for_index(config, args.index)
        failed = run_campaign(config, args.output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
