"""Route/pose privileged teacher validation, independent of the visual actor."""

import argparse
import json

import numpy as np

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.imitation import teacher_action


def evaluate_teacher(config, suite, output, target_speed=2.0):
    run = begin_run(output, {
        "environment": config, "suite": suite, "target_speed": target_speed,
        "teacher_privilege": "route/pose truth; not a vision-driving policy",
    })
    rows = []
    for case in suite["cases"]:
        environment = {**config, "max_steps": int(suite.get("steps", 8000))}
        environment["road"] = {**config.get("road", {}), **case.get("road", {})}
        env = FlyBrainEnv(**environment)
        try:
            _, info = env.reset(seed=case.get("seed", 8001), options={
                key: case[key] for key in ("start_x", "lateral_offset", "heading_offset_degrees")
                if key in case
            })
            initial = info["truth"]["progress_3d"]
            errors, trace = [], []
            for tick in range(environment["max_steps"]):
                action = teacher_action(env, info, target_speed)
                _, _, terminated, truncated, info = env.step(action)
                errors.append(info["truth"]["lateral_error"])
                trace.append({"tick": tick, "action": action.tolist(), **info})
                if terminated or truncated:
                    break
            row = {
                "case": case["name"], "reason": info["reason"], "steps": len(errors),
                "progress_delta": info["truth"]["progress_3d"] - initial,
                "lateral_rms": float(np.sqrt(np.mean(np.array(errors) ** 2))),
                "lateral_max": float(np.max(np.abs(errors))),
                "completion": info["reason"] == "completion",
            }
            rows.append(row)
            write_json(run / f"trace-{len(rows):02d}.json", trace)
            write_json(run / "evaluation.json", {"cases": rows})
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    write_json(run / "result.json", {"status": "completed", "completion_count": sum(
        r["completion"] for r in rows
    )})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, encoding="utf-8"))
    suite = json.load(open(args.suite, encoding="utf-8"))
    evaluate_teacher(config["environment"], suite, args.output)
