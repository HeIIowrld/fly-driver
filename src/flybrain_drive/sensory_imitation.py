"""Sensor-only recurrent BC/DAgger and independent fixed-speed steering evaluation.

Fixed speed is a declared task constraint, never a claim of autonomous speed control.
Route truth is used by the teacher/evaluator, not by the actor's packed observation.
"""

import copy
import hashlib
import time

import numpy as np
import torch

from .env import FlyBrainEnv
from .evidence import begin_run, write_json
from .graph import sha256
from .imitation import fit_sequences, teacher_action
from .learning import graph_for_config, save_checkpoint, select_device
from .policy import policy_from_config
from .sensory_policy import SensoryPolicyRunner
from .sensory_runtime import make_sensory_runtime


def fixed_speed_action(action, env, speed):
    if not np.isfinite(speed) or not 0 < speed < env.vehicle.max_speed:
        raise ValueError("Fixed speed must be positive and below configured max speed")
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Fixed-speed steering currently requires two-intent legacy actuation")
    action[0] = 2 * speed / env.vehicle.max_speed - 1
    return action


def policy_digest(policy):
    digest = hashlib.sha256()
    for key, value in policy.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def checked_environment(config):
    environment = copy.deepcopy(config["environment"])
    if environment.get("mode") != "V" or environment.get("actuation_mode", "legacy") != "legacy":
        raise ValueError("New sensory BC currently requires V mode and legacy two-intent actuation")
    return environment


def collect_sequences(config, settings, output, student=None, teacher_beta=1.0):
    environment = checked_environment(config)
    episodes = settings["episodes"]
    if (not episodes or any(type(e["steps"]) is not int or e["steps"] < 1 for e in episodes)
            or not np.isfinite(teacher_beta) or not 0 <= teacher_beta <= 1
            or (student is None and teacher_beta != 1)):
        raise ValueError("Collection needs positive budgets and a valid teacher mixture")
    speed = float(config["fixed_speed_mps"])
    env = FlyBrainEnv(**environment)
    run = None
    try:
        bridge, packer, effective = make_sensory_runtime(env, config)
        fixed_speed_action(np.zeros(2), env, speed)
        fingerprint = policy_digest(student) if student is not None else None
        run = begin_run(output, {"policy_configuration": effective, "collection_settings": settings,
                                 "teacher_beta": teacher_beta, "student_state_sha256": fingerprint,
                                 "collection": "expert" if student is None else "dagger"})
        write_json(run / "eye-maps.json", {eye.side: eye.manifest() for eye in bridge.eyes})
        runner = SensoryPolicyRunner(student, packer) if student is not None else None
        rng = np.random.default_rng(int(settings["data_seed"]))
        dataset, summaries = [], []
        for index, episode in enumerate(episodes):
            remaining = episode["steps"]
            while remaining:
                seed = int(settings["data_seed"]) + len(dataset)
                obs, info = env.reset(seed=seed, options={k: episode[k] for k in
                    ("start_x", "lateral_offset", "heading_offset_degrees") if k in episode})
                episode_id = f"segment-{len(dataset)}"
                initial = info["truth"]["progress_3d"]
                rows, labels, executed, chosen, stamps, exposure_ids = [], [], [], [], [], []
                for tick in range(remaining):
                    actor = bridge.observe(env, obs, episode_id=episode_id, sequence_id=tick,
                                           received_monotonic_seconds=time.monotonic())
                    label = fixed_speed_action(teacher_action(env, info, speed), env, speed)
                    use_teacher = student is None or bool(rng.random() < teacher_beta)
                    if runner is not None:
                        prediction, packed = runner.step(actor)
                        command = (label if use_teacher
                                   else fixed_speed_action(prediction, env, speed))
                    else:
                        packed, command = packer.pack(actor), label
                    rows.append(packed)
                    labels.append(label.copy())
                    executed.append(command.copy())
                    chosen.append(use_teacher)
                    stamps.append(actor.simulation_seconds)
                    exposure_ids.append([eye.frame_id for eye in actor.eyes])
                    obs, _, terminated, truncated, info = env.step(command)
                    if terminated or truncated:
                        break
                row = {"observation": {key: np.stack([p[key] for p in rows]) for key in rows[0]},
                       "action": np.stack(labels)}
                path = run / f"episode-{len(dataset):03d}.npz"
                np.savez_compressed(path,
                    **{"obs_" + k: v for k, v in row["observation"].items()},
                    action=row["action"], executed_action=np.stack(executed),
                    teacher_selected=np.array(chosen), observation_time=np.array(stamps),
                    exposure_ids=np.array(exposure_ids),
                )
                reason = info["reason"] if not env.active else "collection_segment_budget"
                summaries.append({"file": path.name, "sha256": sha256(path), "seed": seed,
                                  "requested_segment": index, "steps": len(labels),
                                  "teacher_decisions": int(np.sum(chosen)), "reason": reason,
                                  "diagnostic_progress_m": info["truth"]["progress_3d"] - initial})
                dataset.append(row)
                remaining -= len(labels)
                write_json(run / "progress.json", {"segments": summaries})
                if student is None:
                    if reason not in ("running", "completion", "time_limit",
                                      "collection_segment_budget"):
                        raise RuntimeError("Teacher demonstration failed: " + reason)
                    break
        if student is not None and policy_digest(student) != fingerprint:
            raise RuntimeError("Student weights changed while collecting DAgger")
        write_json(run / "dataset.json", {
            "episodes": summaries, "decisions": sum(len(e["action"]) for e in dataset),
            "actor_fields": list(dataset[0]["observation"]),
            "teacher_privilege": "route/pose used only for labels and separate evaluation",
            "student_state_sha256": fingerprint, "teacher_beta": teacher_beta,
            "task": "fixed speed; student controls steering only",
        })
        write_json(run / "result.json", {"status": "completed"})
        return dataset, summaries
    except Exception as error:
        if run is not None:
            write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                             "message": str(error)})
        raise
    finally:
        env.close()


def evaluate_sensory(policy, config, suite, output, *, vision="normal"):
    if vision not in ("normal", "missing"):
        raise ValueError("Unknown vision evaluation condition")
    cases, steps = suite["cases"], int(suite["steps"])
    if (steps < 1 or not cases
            or len({c["name"] for c in cases}) != len(cases)):
        raise ValueError("Evaluation needs unique cases and a positive independent budget")
    run = begin_run(output, {"policy_configuration": config, "suite": suite, "vision": vision,
                             "policy_state_sha256": policy_digest(policy),
                             "task": "fixed speed; no teacher or steering fallback"})
    results = []
    try:
        for case in cases:
            environment = checked_environment(config)
            environment["max_steps"] = steps
            environment["road"] = {**environment.get("road", {}), **case.get("road", {})}
            env = FlyBrainEnv(**environment)
            try:
                obs, info = env.reset(seed=int(case["seed"]), options={k: case[k] for k in
                    ("start_x", "lateral_offset", "heading_offset_degrees") if k in case})
                bridge, packer, _ = make_sensory_runtime(env, config)
                runner = SensoryPolicyRunner(policy, packer)
                initial = info["truth"]["progress_3d"]
                trace = []
                for tick in range(steps):
                    if vision == "missing":
                        obs = {**obs, "camera_valid": np.zeros(2, np.float32)}
                    actor = bridge.observe(env, obs, episode_id=case["name"], sequence_id=tick,
                                           received_monotonic_seconds=time.monotonic())
                    requested, packed = runner.step(actor)
                    command = fixed_speed_action(requested, env, float(config["fixed_speed_mps"]))
                    obs, _, terminated, truncated, info = env.step(command)
                    trace.append({"observation_time": actor.simulation_seconds,
                                  "requested_action": requested.tolist(),
                                  "executed_action": command.tolist(),
                                  "visible_ray_fraction": packed["ray_valid"].mean(1).tolist(),
                                  "diagnostic_progress_m": info["truth"]["progress_3d"],
                                  "diagnostic_lateral_m": info["truth"]["lateral_error"],
                                  "reason": info["reason"]})
                    if terminated or truncated:
                        break
                reason = info["reason"] if not env.active else "evaluation_budget"
                result = {"case": case["name"], "seed": case["seed"], "steps": len(trace),
                          "reason": reason, "completion": reason == "completion",
                          "failure": reason not in (
                              "completion", "time_limit", "evaluation_budget"),
                          "progress_delta": info["truth"]["progress_3d"] - initial,
                          "lateral_error_rms": float(np.sqrt(np.mean(
                              [t["diagnostic_lateral_m"]**2 for t in trace])))}
                results.append(result)
                write_json(run / f"trace-{len(results):02d}.json", trace)
                write_json(run / "progress.json", {"cases": results})
                print(f"sensory_eval={case['name']} progress={result['progress_delta']:.2f} "
                      f"reason={reason}", flush=True)
            finally:
                env.close()
        aggregate = {"cases": len(results),
                     "completion_rate": float(np.mean([r["completion"] for r in results])),
                     "failure_rate": float(np.mean([r["failure"] for r in results])),
                     "mean_progress_delta": float(np.mean([r["progress_delta"] for r in results]))}
        write_json(run / "evaluation.json", {"cases": results, "aggregate": aggregate})
        write_json(run / "result.json", {"status": "completed"})
        return {"cases": results, "aggregate": aggregate}
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                         "message": str(error)})
        raise


def train_sensory(config, output):
    checked_environment(config)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    torch.manual_seed(int(config["seed"]))
    device = select_device(config)
    env = FlyBrainEnv(**config["environment"])
    try:
        _, _, config = make_sensory_runtime(env, config)
        fixed_speed_action(np.zeros(2), env, float(config["fixed_speed_mps"]))
        config["behavior_cloning"]["supervised_action_mask"] = [0, 1]
    finally:
        env.close()
    run = begin_run(output, config)
    try:
        graph, graph_metadata = graph_for_config(config)
        policy = policy_from_config(config, graph).to(device)
        optimizer = torch.optim.Adam(policy.parameters(),
                                      lr=float(config["behavior_cloning"]["learning_rate"]))
        save_checkpoint(run / "random-checkpoint.pt", policy, optimizer, config, graph, 0)
        initial = evaluate_sensory(policy, config, config["evaluation"], run / "random-evaluation")
        settings = copy.deepcopy(config["behavior_cloning"])
        dataset, expert = collect_sequences(config, settings, run / "teacher-data")
        logs = fit_sequences(policy, dataset, settings, optimizer, device, "sensory-expert",
                             run / "bc-progress.json")
        save_checkpoint(run / "bc-checkpoint.pt", policy, optimizer, config, graph, len(logs))
        bc = evaluate_sensory(policy, config, config["evaluation"], run / "bc-evaluation")
        rounds = []
        for index, options in enumerate(settings.get("dagger_rounds", [])):
            if "teacher_beta" not in options or "episodes" not in options:
                raise ValueError("Each DAgger round needs an explicit mixture and decision budget")
            phase = {**settings, **options, "supervised_action_mask": [0, 1]}
            extra, evidence = collect_sequences(config, phase, run / f"dagger-data-{index}",
                                                 student=policy,
                                                 teacher_beta=float(options["teacher_beta"]))
            dataset.extend(extra)
            logs.extend(fit_sequences(policy, dataset, phase, optimizer, device,
                                       f"sensory-dagger-{index}", run / "bc-progress.json"))
            save_checkpoint(run / f"dagger-{index}-checkpoint.pt", policy, optimizer,
                            config, graph, len(logs))
            evaluation = evaluate_sensory(policy, config, config["evaluation"],
                                           run / f"dagger-{index}-evaluation")
            rounds.append({"round": index, "collection": evidence, "evaluation": evaluation,
                           "aggregate_samples": sum(len(e["action"]) for e in dataset)})
        save_checkpoint(run / "checkpoint.pt", policy, optimizer, config, graph, len(logs))
        summary = {"task": "fixed speed steering, not autonomous speed/AC/full fly circuit",
                   "graph": graph_metadata, "random": initial, "bc": bc,
                   "dagger_rounds": rounds, "expert": expert, "training_epochs": logs,
                   "final": rounds[-1]["evaluation"] if rounds else bc}
        if config.get("evaluate_vision_missing", False):
            summary["vision_missing"] = evaluate_sensory(
                policy, config, config["evaluation"], run / "vision-missing-evaluation",
                vision="missing",
            )
        write_json(run / "summary.json", summary)
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False,
                                         "checkpoint_sha256": sha256(run / "checkpoint.pt")})
        return run
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                         "message": str(error)})
        raise
