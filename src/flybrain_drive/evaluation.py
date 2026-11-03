"""Paired deterministic evaluations with an independent, declared episode budget."""

import hashlib

import numpy as np
import torch

from .env import FlyBrainEnv
from .evidence import begin_run, write_json
from .graph import sha256
from .learning import restore_policy
from .policy import tensor_observation
from .replay import PolicyReplay


def evaluate_suite(checkpoint, output, suite, record=False, record_every=4):
    steps = int(suite.get("steps", 4000))
    cases = suite["cases"]
    names = [case["name"] for case in cases]
    if steps < 1 or not names or len(names) != len(set(names)):
        raise ValueError("Evaluation needs a positive budget and unique case names")
    if record and record_every < 1:
        raise ValueError("record_every must be positive")
    policy, config = restore_policy(checkpoint)
    run = begin_run(
        output,
        {
            "checkpoint_sha256": sha256(checkpoint),
            "suite": suite,
            "training_config": config,
            "evaluation_device": "cpu",
            "action_selection": "tanh(mean), deterministic",
            "record": record,
            "record_every": record_every if record else None,
        },
    )
    results = []
    try:
        for case in cases:
            # Case overrides contain road geometry only; actor sensing/mode is unchanged.
            environment = {**config.get("environment", {}), "max_steps": steps}
            environment["road"] = {
                **environment.get("road", {}),
                **case.get("road", {}),
            }
            env = FlyBrainEnv(**environment)
            replay = None
            try:
                obs, info = env.reset(
                    seed=int(case.get("seed", 8001)),
                    options={
                        "start_x": float(case.get("start_x", 3.0)),
                        "lateral_offset": float(case.get("lateral_offset", 0)),
                        "heading_offset_degrees": float(case.get("heading_offset_degrees", 0)),
                    },
                )
                initial_progress = info["truth"]["progress_3d"]
                initial_time = info["time"]
                remaining = max(env.road.s[-1] - 1 - initial_progress, 1e-6)
                state = policy.initial_state()
                trace, reward_sum = [], 0.0
                if record:
                    replay = PolicyReplay(
                        env, run / f"recording-{len(results) + 1:02d}", case["name"], record_every
                    )
                for tick in range(steps):
                    with torch.no_grad():
                        mean, _, state = policy(tensor_observation(obs), state)
                    action = torch.tanh(mean)[0].numpy()
                    input_observation, input_info = obs, info
                    frame = replay.capture(tick, obs, info, action) if replay else None
                    obs, reward, terminated, truncated, info = env.step(action)
                    if replay:
                        replay.after_step(frame, info)
                    reward_sum += reward
                    rotation = env.data.xmat[env.model.body("chassis").id].reshape(3, 3)
                    up = np.asarray(info["truth"]["road_frame"])[:, 2]
                    tilt = np.degrees(np.arccos(np.clip(np.dot(rotation[:, 2], up), -1, 1)))
                    trace.append(
                        {
                            "tick": tick,
                            "action": action.tolist(),
                            "observation_time": input_info["time"],
                            "transition_end_time": info["time"],
                            "body_at_decision": input_observation["body"].tolist(),
                            "body_valid_at_decision": input_observation["body_valid"].tolist(),
                            "camera_age_at_decision": input_observation["camera_age"].tolist(),
                            "camera_valid_at_decision": input_observation["camera_valid"].tolist(),
                            "image_sha256_at_decision": hashlib.sha256(
                                input_observation["images"].tobytes()
                            ).hexdigest(),
                            "issued_control": info["control"],
                            "applied_actuator_control": env.data.ctrl.tolist(),
                            "reward": reward,
                            "progress_3d": info["truth"]["progress_3d"],
                            "lateral_error": info["truth"]["lateral_error"],
                            "position_xyz": info["truth"]["position_xyz"],
                            "road_relative_tilt_degrees": float(tilt),
                            "reason": info["reason"],
                        }
                    )
                    if terminated or truncated:
                        break
                lateral = np.array([row["lateral_error"] for row in trace])
                progress = info["truth"]["progress_3d"] - initial_progress
                reason = info["reason"] if not env.active else "evaluation_budget"
                results.append(
                    {
                        "case": case["name"],
                        "seed": int(case.get("seed", 8001)),
                        "start_x": float(case.get("start_x", 3.0)),
                        "environment": env.configuration(),
                        "steps": len(trace),
                        "simulated_seconds": info["time"] - initial_time,
                        "reward_sum": reward_sum,
                        "reason": reason,
                        "completion": reason == "completion",
                        "failure": reason not in ("completion", "time_limit", "evaluation_budget"),
                        "initial_progress_3d": initial_progress,
                        "progress_delta": progress,
                        "remaining_course_fraction": float(np.clip(progress / remaining, 0, 1)),
                        "lateral_error_rms": float(np.sqrt(np.mean(lateral**2))),
                        "lateral_error_max": float(np.max(np.abs(lateral))),
                        "road_relative_tilt_max_degrees": max(
                            row["road_relative_tilt_degrees"] for row in trace
                        ),
                    }
                )
                if replay:
                    replay.capture(len(trace), obs, info, final=True)
                    results[-1]["recording"] = replay.finish()
                write_json(run / f"trace-{len(results):02d}.json", trace)
                write_json(run / "progress.json", {"completed_cases": results})
                print(
                    f"eval={case['name']} steps={len(trace)} "
                    f"progress={progress:.2f} reason={reason}",
                    flush=True,
                )
            finally:
                if replay:
                    replay.close()
                env.close()
        aggregate = {
            "cases": len(results),
            "completion_rate": float(np.mean([r["completion"] for r in results])),
            "failure_rate": float(np.mean([r["failure"] for r in results])),
            "mean_progress_delta": float(np.mean([r["progress_delta"] for r in results])),
            "mean_remaining_course_fraction": float(
                np.mean([r["remaining_course_fraction"] for r in results])
            ),
            "mean_lateral_error_rms": float(np.mean([r["lateral_error_rms"] for r in results])),
        }
        write_json(run / "evaluation.json", {"cases": results, "aggregate": aggregate})
        write_json(run / "result.json", {"status": "completed"})
        return {"cases": results, "aggregate": aggregate}
    except Exception as error:
        write_json(
            run / "result.json",
            {
                "status": "failed",
                "reason": type(error).__name__,
                "message": str(error),
            },
        )
        raise
