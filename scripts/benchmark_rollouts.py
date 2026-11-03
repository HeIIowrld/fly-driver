"""Fixed-input CPU/render benchmark, profile and exact replay evidence."""

import argparse
import cProfile
import hashlib
import json
import pstats
import time
from pathlib import Path

import numpy as np

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evidence import begin_run, write_json


def benchmark(output, steps=300, repeats=3):
    run = begin_run(
        output,
        {
            "steps": steps,
            "repeats": repeats,
            "action": [-0.75, 0.0],
            "scope": "local CPU/renderer throughput; not CUDA learning speed",
        },
    )
    rows = []
    for mode in ("O", "V"):
        for repeat in range(repeats):
            env = FlyBrainEnv(mode=mode, max_steps=steps + 1)
            profile = cProfile.Profile()
            digest = hashlib.sha256()
            trace = []
            durations = []
            try:
                obs, _ = env.reset(seed=901)
                # Warm renderer/library initialization separately from measured steps.
                env.reset(seed=901)
                for _ in range(steps):
                    started = time.perf_counter()
                    if repeat == 0:
                        profile.enable()
                    obs, reward, terminated, truncated, info = env.step(np.array([-0.75, 0.0]))
                    if repeat == 0:
                        profile.disable()
                    durations.append(time.perf_counter() - started)
                    for key in sorted(obs):
                        digest.update(key.encode())
                        digest.update(obs[key].tobytes())
                    digest.update(env.data.qpos.tobytes())
                    digest.update(env.data.qvel.tobytes())
                    digest.update(np.array([reward], dtype=np.float64).tobytes())
                    digest.update(info["reason"].encode())
                    trace.append(
                        [info["truth"]["progress_3d"], info["truth"]["lateral_error"], reward]
                    )
                    if terminated or truncated:
                        break
                elapsed = sum(durations)
                row = {
                    "mode": mode,
                    "repeat": repeat,
                    "steps": len(trace),
                    "step_seconds": elapsed,
                    "steps_per_second": len(trace) / elapsed,
                    "p50_p95_ms": (np.quantile(durations, [0.5, 0.95]) * 1000).tolist(),
                    "exact_replay_sha256": digest.hexdigest(),
                    "reason": info["reason"],
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_json(run / "progress.json", rows)
                if repeat == 0:
                    write_json(run / f"trace-{mode}.json", trace)
                    with (run / f"profile-{mode}.txt").open("w") as stream:
                        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(
                            25
                        )
            finally:
                env.close()
    write_json(
        run / "benchmark.json",
        {
            "rows": rows,
            "unprofiled_median_sps": {
                mode: float(
                    np.median(
                        [
                            row["steps_per_second"]
                            for row in rows
                            if row["mode"] == mode and row["repeat"] > 0
                        ]
                    )
                )
                for mode in ("O", "V")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.repeats < 2:
        raise ValueError("Need positive steps and at least two repeats")
    benchmark(arguments.output, arguments.steps, arguments.repeats)
