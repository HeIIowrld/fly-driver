"""Compare real PPO CPU/GPU throughput and worker counts before local pilots."""

import argparse
import faulthandler
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from scripts.run_campaign import merged


def benchmark(config, output):
    from flybrain_drive.parallel_learning import train_parallel

    faulthandler.dump_traceback_later(90, repeat=True)
    run = begin_run(output, config)
    rows = []
    for device, count in (("cpu", 4), ("cuda:0", 4), ("cuda:0", 8)):
        trial = merged(config, config.get("campaign", {}).get("arms", [{}])[0].get("overrides", {}))
        trial.pop("campaign", None)
        trial.pop("behavior_cloning", None)
        trial.pop("local_screening", None)
        trial.update(device=device, num_envs=count, horizon=128, updates=3, epochs=2,
                     checkpoint_interval=3)
        out = run / f"{device.replace(':', '-')}-{count}"
        try:
            train_parallel(trial, out)
            training = json.loads((out / "training.json").read_text())
            warm = training["updates"][1:]
            steps = count * 128 * len(warm)
            seconds = sum(row["update_wall_seconds"] for row in warm)
            row = {
                "device": device, "num_envs": count, "status": "completed",
                "warm_transitions_per_second": steps / seconds,
                "startup_seconds": training["startup_and_warmstart_wall_seconds"],
                "first_update_seconds": training["updates"][0]["update_wall_seconds"],
                "warm_policy_seconds": sum(r["policy_seconds"] for r in warm),
                "warm_environment_seconds": sum(r["environment_seconds"] for r in warm),
                "warm_optimizer_seconds": sum(r["optimizer_seconds"] for r in warm),
                "parameter_change_l2": training["parameter_change_l2"],
                "device_evidence": json.loads((out / "device.json").read_text()),
                "worker_runtime": [worker["worker_runtime"] for worker in
                                   json.loads((out / "environment.json").read_text())["workers"]],
            }
        except Exception as error:
            row = {"device": device, "num_envs": count, "status": "failed",
                   "reason": type(error).__name__, "message": str(error)}
        rows.append(row)
        write_json(run / "progress.json", rows)
        print(json.dumps(row), flush=True)
    gpu = [r for r in rows if r["status"] == "completed" and r["device"] == "cuda:0"]
    best = max(gpu, key=lambda r: r["warm_transitions_per_second"]) if gpu else None
    write_json(run / "result.json", {
        "rows": rows, "recommended_gpu_workers": best["num_envs"] if best else None,
        "scope": "3 short updates/setting, first excluded; throughput diagnostic",
        "gpu_sps_spread": float(np.ptp([r["warm_transitions_per_second"] for r in gpu]))
        if gpu else None,
    })
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(json.loads(args.config.read_text()), args.output)
