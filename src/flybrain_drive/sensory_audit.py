"""Privileged teacher diagnostics, kept outside sensor-only policy inference."""

import numpy as np

from .env import FlyBrainEnv
from .evidence import begin_run, write_json
from .graph import sha256
from .imitation import teacher_action
from .sensory_imitation import checked_environment, fixed_speed_action


def evaluate_teacher(config, suite, output, *, render=True):
    """Run the same V environment and declared fixed-speed execution as the student.

    This is an oracle controller check, never an autonomous vision-agent result.
    Every saved diagnostic row pairs pre-step truth/command with post-step truth.
    """
    cases, steps = suite["cases"], suite["steps"]
    if (type(steps) is not int or steps < 1 or not cases
            or len({c["name"] for c in cases}) != len(cases)):
        raise ValueError("Teacher evaluation needs unique cases and a positive integer budget")
    run = begin_run(output, {"policy_configuration": config, "suite": suite,
                             "render": render,
                             "controller": "privileged route teacher, not vision policy"})
    results = []
    try:
        for index, case in enumerate(cases):
            environment = checked_environment(config)
            # Teacher consumes truth in either mode; O omits costly image rendering.
            # This switch is for oracle diagnostics only, never student evaluation.
            if not render:
                environment["mode"] = "O"
            environment["max_steps"] = steps
            environment["road"] = {**environment.get("road", {}), **case.get("road", {})}
            env = FlyBrainEnv(**environment)
            try:
                _, info = env.reset(seed=int(case["seed"]), options={k: case[k] for k in
                    ("start_x", "lateral_offset", "heading_offset_degrees") if k in case})
                initial = info["truth"]["progress_3d"]
                rows = []
                for tick in range(steps):
                    command = fixed_speed_action(
                        teacher_action(env, info, config["fixed_speed_mps"]),
                        env, config["fixed_speed_mps"],
                    )
                    pre = [info["time"], info["truth"]["progress_3d"],
                           info["truth"]["lateral_error"]]
                    _, _, terminated, truncated, info = env.step(command)
                    rows.append(pre + command.tolist() + [info["time"],
                        info["truth"]["progress_3d"], info["truth"]["lateral_error"]])
                    if tick % 1000 == 0:
                        write_json(run / "heartbeat.json", {
                            "case": case["name"], "decision": tick + 1,
                            "simulation_seconds": info["time"],
                        })
                    if terminated or truncated:
                        break
                trace = np.asarray(rows)
                path = run / f"teacher-trace-{index:02d}.npz"
                np.savez_compressed(path, trace=trace, columns=np.array([
                    "pre_time", "pre_progress", "pre_lateral", "speed_intent", "steer_intent",
                    "post_time", "post_progress", "post_lateral",
                ]))
                reason = info["reason"] if not env.active else "evaluation_budget"
                result = {
                    "case": case["name"], "seed": case["seed"], "steps": len(rows),
                    "reason": reason, "completion": reason == "completion",
                    "progress_delta": info["truth"]["progress_3d"] - initial,
                    "lateral_error_rms": float(np.sqrt(np.mean(trace[:, 7]**2))),
                    "steering_saturation_fraction": float(np.mean(np.abs(trace[:, 4]) >= .99)),
                    "trace_file": path.name, "trace_sha256": sha256(path),
                }
                results.append(result)
                write_json(run / "progress.json", {"cases": results})
                print(f"teacher_eval={case['name']} reason={reason} "
                      f"progress={result['progress_delta']:.2f}", flush=True)
            finally:
                env.close()
        summary = {"controller": "privileged teacher, not autonomous sensor driving",
                   "cases": results, "completion_rate": float(np.mean([
                       r["completion"] for r in results]))}
        write_json(run / "evaluation.json", summary)
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False})
        return summary
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                         "message": str(error)})
        raise
