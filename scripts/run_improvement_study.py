"""Bounded ROCm screening, paired replication, and a sealed promotion proposal."""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json


def paired_gate(baseline, candidate):
    if baseline.get("status") != "completed" or candidate.get("status") != "completed":
        return {"passed": False, "reason": "missing_or_failed_run"}
    if baseline["seed"] != candidate["seed"]:
        raise ValueError("Screen comparison must use the same training seed")
    a, b = baseline["final"], candidate["final"]
    if [c["case"] for c in a["cases"]] != [c["case"] for c in b["cases"]]:
        raise ValueError("Screen comparison case order differs")
    improved = sum(cb["progress_delta"] > ca["progress_delta"]
                   for ca, cb in zip(a["cases"], b["cases"], strict=True))
    before = candidate["initial"]["aggregate"]["mean_progress_delta"]
    final = b["aggregate"]["mean_progress_delta"]
    control = a["aggregate"]["mean_progress_delta"]
    retention = final / max(before, 1e-6)
    relative = (final - control) / max(control, 1e-6)
    accepted = sum(row.get("accepted_optimizer_steps", row["epochs_ran"])
                   for row in candidate["training"]["updates"])
    passed = (
        b["aggregate"]["failure_rate"] <= a["aggregate"]["failure_rate"]
        and relative >= .10 and improved >= 3 and retention >= .90 and accepted > 0
    )
    return {"passed": passed, "improved_cases": improved, "relative_last_progress_gain": relative,
            "last_vs_initial_progress_ratio": retention, "accepted_optimizer_steps": accepted,
            "baseline_final_m": control, "candidate_initial_m": before, "candidate_final_m": final,
            "interpretation": "last checkpoint screen; selected-BC fallback cannot win this gate"}


def finish_child(process, timeout):
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        return 124


def trial(config, run, arm, seed, phase):
    settings = config["improvement_study"]
    task = f"{phase}-{arm['name']}-{seed}"
    trial_config = copy.deepcopy(config)
    trial_config.pop("improvement_study")
    trial_config["campaign"]["arms"] = [arm]
    trial_config["campaign"]["seeds"] = [seed]
    config_path = run / f"{task}-input.json"
    write_json(config_path, trial_config)
    output = run / task
    with (run / f"{task}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "scripts.run_campaign", "--config", str(config_path),
             "--output", str(output)], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(run / "live.json", {"phase": phase, "arm": arm["name"], "seed": seed,
                                      "owned_process_group": process.pid,
                                      "deadline_seconds": settings["trial_timeout_seconds"]})
        code = finish_child(process, int(settings["trial_timeout_seconds"]))
    summary = output / "summary.json"
    if summary.exists():
        row = json.loads(summary.read_text(encoding="utf-8"))["runs"][0]
    else:
        row = {"arm": arm["name"], "seed": seed, "status": "failed",
               "reason": "timeout" if code == 124 else "missing_summary", "exit_code": code}
    return {**row, "study_phase": phase, "trial_output": str(output), "process_exit_code": code}


def remote_proposal(config, selected):
    settings = config["improvement_study"]
    result = copy.deepcopy(config)
    result.pop("improvement_study")
    result["behavior_cloning"].pop("precollected_teacher", None)
    # Every arm receives a fresh remote model, expert data and student-visited DAgger data.
    result.update(num_envs=8, torch_threads=2, updates=128,
                  retain_checkpoint_interval=16)
    result["validation"].update(interval=16)
    result["campaign"].update(
        seeds=settings["remote_seeds"], evaluation=settings["locked_test"],
        arms=[config["campaign"]["arms"][0], *selected], evaluation_role="locked_test",
    )
    for arm in result["campaign"]["arms"]:
        arm.get("overrides", {}).get("behavior_cloning", {}).pop("precollected_teacher", None)
    result["promotion"] = {
        "basis": "paired local last-checkpoint gates; two fresh development seeds",
        "selected": [a["name"] for a in selected], "no_local_checkpoint_transfer": True,
    }
    return result


def run_study(config, output):
    if os.name != "posix":
        raise RuntimeError("Local study runs inside the verified WSL ROCm environment")
    from scripts.benchmark_acceleration import benchmark

    run = begin_run(output, config)
    settings = config["improvement_study"]
    rows, gates = [], []
    try:
        benchmark(config, run / "renderer-gate", timeout_seconds=180)
        gate = json.loads((run / "renderer-gate/result.json").read_text(encoding="utf-8"))
        hardware = next(c for c in gate["cases"] if c["renderer"] == "d3d12")
        renderer = hardware.get("profile", {}).get("graphics", {}).get("renderer", "")
        if (hardware["status"] != "completed" or "AMD Radeon RX 9070 XT" not in renderer
                or not gate["fixed_command_equivalence"].get("physics_equal", False)):
            raise RuntimeError("D3D12 hardware-camera/PPO/equivalent-physics gate failed")
        arms = config["campaign"]["arms"]
        smoke = copy.deepcopy(config)
        smoke.pop("behavior_cloning", None)
        smoke.update(horizon=64, updates=3, retain_checkpoint_interval=1)
        smoke["improvement_study"]["trial_timeout_seconds"] = 180
        smoke["validation"].update(interval=1, record=False)
        smoke["validation"]["suite"] = {
            "steps": 32, "cases": [{"name": "smoke_dev", "seed": 9398}],
        }
        smoke["campaign"].update(record_evaluation=False, evaluation={
            "steps": 32, "cases": [{"name": "smoke_eval", "seed": 9498}],
        })
        feature_gate = trial(smoke, run, arms[-1], 7599, "feature_gate")
        write_json(run / "feature-gate.json", feature_gate)
        if feature_gate["status"] != "completed":
            raise RuntimeError("GPU anchor/critic/checkpoint/validation gate failed")
        seed, replication_seed = settings["local_seeds"]
        def execute(arm, seed, phase):
            if len(rows) >= int(settings["max_local_trials"]):
                raise ValueError("Local study exceeded its sealed trial limit")
            row = trial(config, run, arm, seed, phase)
            rows.append(row)
            write_json(run / "summary.json", {"runs": rows, "gates": gates})
            return row
        baseline = execute(arms[0], seed, "screen")
        passing = []
        for arm in arms[1:]:
            row = execute(arm, seed, "screen")
            decision = {"arm": arm["name"], "seed": seed, **paired_gate(baseline, row)}
            gates.append(decision)
            if decision["passed"]:
                passing.append(arm)
        if not passing:
            fallback = settings["recovery_fallback"]
            row = execute(fallback, seed, "screen_recovery")
            decision = {"arm": fallback["name"], "seed": seed, **paired_gate(baseline, row)}
            gates.append(decision)
            if decision["passed"]:
                passing.append(fallback)
        replicated = []
        if passing:
            baseline_replication = execute(arms[0], replication_seed, "replication")
            for arm in passing[:2]:
                row = execute(arm, replication_seed, "replication")
                decision = {"arm": arm["name"], "seed": replication_seed,
                            **paired_gate(baseline_replication, row)}
                gates.append(decision)
                if decision["passed"]:
                    replicated.append(arm)
        proposal = remote_proposal(config, replicated) if replicated else None
        if proposal is not None:
            write_json(run / "remote-proposal.json", proposal)
        write_json(run / "summary.json", {"runs": rows, "gates": gates})
        write_json(run / "result.json", {
            "status": "completed", "selected": [a["name"] for a in replicated],
            "local_trials": len(rows),
            "failed_trials": sum(r["status"] != "completed" for r in rows),
            "remote_proposal": bool(proposal),
            "physics": "synthetic 3D course; no hardware control",
            "next_stage": ("remote_replication" if proposal
                           else "review_local_failures_and_recovery"),
        })
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                        "message": str(error)})
        raise
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_study(json.loads(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
