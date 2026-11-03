"""Privileged geometric teacher; sensor-only sequence BC with explicit episode boundaries."""

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .env import FlyBrainEnv
from .episode_storage import load_array
from .evidence import begin_run, write_json
from .graph import sha256
from .policy import tensor_observation


def teacher_action(env, info, target_speed=2.0):
    """Teacher-only route and pose truth. Never appended to actor observations."""
    speed = float(np.mean(env._joint_velocities(env.spin_joints)[:2]) * env.vehicle.wheel_radius)
    lookahead = 2.0 + 0.8 * abs(speed)
    progress = min(info["truth"]["progress_3d"] + lookahead, env.road.s[-1])
    point = np.array([np.interp(progress, env.road.s, env.road.xyz[:, i]) for i in range(3)])
    rotation = env.data.xmat[env.model.body("chassis").id].reshape(3, 3)
    local = rotation.T @ (point - env.data.qpos[:3])
    curvature = np.clip(
        2 * local[1] / max(np.sum(local[:2] ** 2), 0.1),
        -env.vehicle.max_curvature, env.vehicle.max_curvature,
    )
    velocity = min(target_speed, env.vehicle.max_speed, np.sqrt(1.5 / max(abs(curvature), 1e-6)))
    action = np.zeros(env.control.action_size, np.float32)
    action[:2] = [2 * velocity / env.vehicle.max_speed - 1, curvature / env.vehicle.max_curvature]
    return action


def collect_teacher_sequences(environment, settings, output, student=None, teacher_beta=1.0):
    """Expert rollouts, or DAgger labels at mixed expert/student states.

    DAgger budgets count decisions, including failed episodes. Episode boundaries
    split the saved sequences and reset the student's numerical recurrent state.
    """
    if environment.get("mode", "O") != "V":
        raise ValueError("Visual BC requires V sensor observations")
    episodes = settings.get("episodes", [{"start_x": 3.0, "steps": 1600}])
    if not episodes or any(int(ep["steps"]) < 1 for ep in episodes):
        raise ValueError("Teacher needs positive episode budgets")
    if not 0 <= teacher_beta <= 1 or (student is None and teacher_beta != 1):
        raise ValueError("DAgger teacher_beta must be between zero and one with a student")
    student_hash = None
    if student is not None:
        digest = hashlib.sha256()
        for key, value in student.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        student_hash = digest.hexdigest()
    run = begin_run(output, {
        "environment": environment, "teacher_settings": settings,
        "collection": "expert" if student is None else "dagger",
        "teacher_beta": teacher_beta, "student_state_sha256": student_hash,
    })
    dataset, summaries = [], []
    teacher_environment = {**environment}
    if settings.get("teacher_actuation_mode"):
        teacher_environment["actuation_mode"] = settings["teacher_actuation_mode"]
    if (teacher_environment.get("actuation_mode", "legacy") == "legacy") != (
        environment.get("actuation_mode", "legacy") == "legacy"
    ):
        raise ValueError("Teacher/actor action schemas must agree")
    env = FlyBrainEnv(**teacher_environment)
    random = np.random.default_rng(int(settings.get("data_seed", 9101)))
    decision_count = 0
    try:
        for index, episode in enumerate(episodes):
            remaining = int(episode["steps"])
            while remaining:
                obs, info = env.reset(
                    seed=int(settings.get("data_seed", 9101)) + len(dataset),
                    options={key: episode[key] for key in (
                        "start_x", "lateral_offset", "heading_offset_degrees"
                    ) if key in episode},
                )
                state = None if student is None else student.initial_state()
                start = info["truth"]["progress_3d"]
                observations, actions, executed, selected, times = [], [], [], [], []
                for _ in range(remaining):
                    action = teacher_action(env, info, float(settings.get("target_speed", 2.0)))
                    use_teacher = True
                    command = action.copy()
                    if student is not None:
                        with torch.no_grad():
                            mean, _, state = student(
                                tensor_observation(obs, student.log_std.device), state
                            )
                        student_action = torch.tanh(mean[0]).cpu().numpy()
                        # A/B/C use the same canonical fixed-TV collection dynamics.
                        # Only speed/curvature are relabelled; yaw starts learning in PPO.
                        if not settings.get("train_yaw_action", True) and len(action) == 3:
                            student_action[2] = 0
                        use_teacher = bool(random.random() < teacher_beta)
                        if not use_teacher:
                            command = student_action
                    observations.append({key: value.copy() for key, value in obs.items()})
                    actions.append(action)
                    executed.append(command.copy())
                    selected.append(use_teacher)
                    times.append(info["time"])
                    obs, _, terminated, truncated, info = env.step(command)
                    if terminated or truncated:
                        break
                row = {
                    "observation": {
                        key: np.stack([item[key] for item in observations])
                        for key in observations[0]
                    },
                    "action": np.stack(actions),
                }
                path = run / f"episode-{len(dataset):02d}.npz"
                np.savez_compressed(
                    path, **{"obs_" + key: value for key, value in row["observation"].items()},
                    action=row["action"], observation_time=np.array(times),
                    executed_action=np.stack(executed), teacher_selected=np.array(selected),
                )
                dataset.append(row)
                summaries.append({
                    "episode": len(dataset) - 1, "requested_segment": index,
                    "steps": len(actions), "start_x": episode.get("start_x", 3.0),
                    "lateral_offset": episode.get("lateral_offset", 0),
                    "heading_offset_degrees": episode.get("heading_offset_degrees", 0),
                    "reason": info["reason"] if not env.active else "collection_segment_budget",
                    "progress_delta": info["truth"]["progress_3d"] - start,
                    "teacher_decisions": int(np.sum(selected)),
                    "dataset_sha256": sha256(path),
                })
                decision_count += len(actions)
                remaining -= len(actions)
                if student is None:
                    if info["reason"] not in ("running", "completion", "time_limit"):
                        raise RuntimeError(
                            "Teacher failed the demonstration gate: " + info["reason"]
                        )
                    break  # Expert completion can end a requested segment early.
        write_json(run / "dataset.json", {
            "episodes": summaries,
            "teacher_privilege": "route centerline and chassis pose for labels only",
            "teacher_actuation_mode": env.control.actuation_mode,
            "collection": "expert" if student is None else "dagger",
            "teacher_beta": teacher_beta, "student_state_sha256": student_hash,
            "decisions": decision_count,
            "actor_fields": list(dataset[0]["observation"]),
            "temporal_contract": "observation/action share decision time; no post-action images",
        })
        write_json(run / "result.json", {"status": "completed"})
        return dataset, summaries
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "reason": str(error)})
        raise
    finally:
        env.close()


def sequence_chunks(episodes, chunk, device):
    """Pad T×episode batches; every real observation has one supervised loss term."""
    longest = max(len(episode["action"]) for episode in episodes)
    count = len(episodes)
    for start in range(0, longest, chunk):
        length = min(chunk, longest - start)
        fields = episodes[0]["observation"]
        observations = {
            key: np.zeros((length, count, *value.shape[1:]), dtype=value.dtype)
            for key, value in fields.items()
        }
        actions = np.zeros((length, count, episodes[0]["action"].shape[1]), np.float32)
        valid = np.zeros((length, count), np.float32)
        for column, episode in enumerate(episodes):
            size = max(0, min(length, len(episode["action"]) - start))
            if size:
                valid[:size, column] = 1
                actions[:size, column] = episode["action"][start:start + size]
                for key in observations:
                    observations[key][:size, column] = (
                        episode["observation"][key][start:start + size]
                    )
        reset = 1 - valid
        if start == 0:
            reset[0] = 1
        yield (
            {key: torch.as_tensor(value, device=device) for key, value in observations.items()},
            torch.as_tensor(actions, device=device), torch.as_tensor(reset, device=device),
            torch.as_tensor(valid, device=device),
        )


def reuse_teacher_sequences(environment, settings, output):
    """Verified expert-only reuse; command ablation is a declared sensor transform."""
    source = Path(settings["precollected_teacher"])
    manifest = json.loads((source / "manifest.json").read_text())
    metadata = json.loads((source / "dataset.json").read_text())
    if metadata["collection"] != "expert":
        raise ValueError("Only policy-independent expert data may be reused")
    source_env = dict(manifest["config"]["environment"])
    target_env = dict(environment)
    source_profile = source_env.pop("sensor_profile", "full")
    target_profile = target_env.pop("sensor_profile", "full")
    if source_env != target_env or source_profile != "full" or target_profile not in (
        "full", "no_efference_copy"
    ):
        raise ValueError("Precollected teacher environment is incompatible")
    original = manifest["config"]["teacher_settings"]
    for name in ("numpy", "mujoco"):
        if manifest["versions"][name] != importlib.metadata.version(name):
            raise ValueError("Precollected teacher runtime version differs: " + name)
    for key in ("data_seed", "target_speed", "episodes", "teacher_actuation_mode"):
        if original.get(key) != settings.get(key):
            raise ValueError("Precollected teacher settings differ: " + key)
    root = Path(__file__).resolve().parents[2]
    for name in ("env.py", "road.py", "control.py", "vision.py"):
        key = "src/flybrain_drive/" + name
        if sha256(root / key) != manifest["source_sha256"][key]:
            raise ValueError("Teacher physics/sensor source changed: " + name)
    run = begin_run(output, {"environment": environment, "teacher_settings": settings,
                             "reuse_source_manifest_sha256": sha256(source / "manifest.json"),
                             "source_profile": source_profile, "target_profile": target_profile})
    dataset, summaries = [], []
    for episode in metadata["episodes"]:
        name = f"episode-{episode['episode']:02d}.npz"
        path = source / name
        if sha256(path) != episode["dataset_sha256"]:
            raise ValueError("Precollected teacher hash mismatch: " + name)
        with load_array(path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        if arrays["obs_body_valid"][:, -6:].any() or arrays["obs_body"][:, -6:].any():
            raise ValueError("Oracle channels found in cached teacher actor input")
        if target_profile == "no_efference_copy":
            arrays["obs_body"][:, 20:24] = 0
            arrays["obs_body_valid"][:, 20:24] = 0
            arrays["obs_previous_yaw_action"][:] = 0
            np.savez_compressed(run / name, **arrays)
        else:
            shutil.copy2(path, run / name)
        dataset.append({"observation": {key[4:]: value for key, value in arrays.items()
                                        if key.startswith("obs_")}, "action": arrays["action"]})
        summaries.append({**episode, "source_dataset_sha256": episode["dataset_sha256"],
                          "dataset_sha256": sha256(run / name)})
    shutil.copy2(source / "manifest.json", run / "source-manifest.json")
    write_json(run / "dataset.json", {**metadata, "episodes": summaries,
                                      "reuse": "verified expert; declared history masking",
                                      "source_manifest_sha256": sha256(source / "manifest.json")})
    write_json(run / "result.json", {"status": "completed", "reason": "verified_expert_reuse"})
    return dataset, summaries


def fit_sequences(policy, dataset, settings, optimizer, device, phase, progress_path=None):
    epochs, chunk = int(settings.get("epochs", 4)), int(settings.get("chunk_length", 128))
    if min(epochs, chunk) < 1:
        raise ValueError("BC epochs and chunk_length must be positive")
    batching = settings.get("batching", "sequential_episodes")
    if batching not in ("sequential_episodes", "parallel_episodes"):
        raise ValueError("Unknown BC batching method")
    # Keep historical pilot settings reproducible. New pilots balance episode
    # columns within each update instead of visiting complete demos in fixed order.
    batch_size = int(settings.get("episodes_per_batch", 12))
    if batch_size < 1:
        raise ValueError("episodes_per_batch must be positive")
    supervised_mask = policy.action_mask.clone()
    if "supervised_action_mask" in settings:
        requested = torch.as_tensor(settings["supervised_action_mask"], device=device)
        if (requested.shape != supervised_mask.shape or not torch.isfinite(requested).all()
                or not torch.all((requested == 0) | (requested == 1))):
            raise ValueError("Supervised action mask must contain one zero/one per action")
        supervised_mask *= requested
    if not settings.get("train_yaw_action", True) and len(supervised_mask) == 3:
        supervised_mask[2] = 0
    if not supervised_mask.any():
        raise ValueError("At least one active action must be supervised")
    logs = []
    for epoch in range(epochs):
        total_loss, count, updates = 0.0, 0, 0
        if batching == "parallel_episodes":
            order = np.random.default_rng(int(settings.get("data_seed", 9101)) + epoch)
            episodes = [dataset[index] for index in order.permutation(len(dataset))]
            groups = [episodes[start:start + batch_size]
                      for start in range(0, len(episodes), batch_size)]
        else:
            groups = [[episode] for episode in dataset]
        for episodes in groups:
            state = policy.initial_state(len(episodes))
            for tensors, target, reset, valid in sequence_chunks(episodes, chunk, device):
                boundary = state.detach().clone()
                mean, _, _ = policy.sequence_batch(tensors, reset, boundary, return_state=True)
                squared = (torch.tanh(mean) - target).square() * supervised_mask * valid[..., None]
                samples = int(valid.sum().item())
                loss = squared.sum() / (samples * supervised_mask.sum())
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite BC loss")
                optimizer.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise RuntimeError("Nonfinite BC gradient")
                optimizer.step()
                with torch.no_grad():
                    _, _, state = policy.sequence_batch(
                        tensors, reset, boundary, return_state=True
                    )
                total_loss += float(loss.detach()) * samples
                count += samples
                updates += 1
        row = {"phase": phase, "epoch": epoch, "mean_action_mse": total_loss / count,
               "supervised_samples": count, "optimizer_updates": updates,
               "gpu_allocated_bytes": torch.cuda.memory_allocated(device)
               if device.type == "cuda" else 0,
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
               if device.type == "cuda" else 0}
        logs.append(row)
        if progress_path:
            write_json(progress_path, row)
        print(f"bc_phase={phase} epoch={epoch} samples={count} "
              f"mse={row['mean_action_mse']:.6f}", flush=True)
    return logs


def teacher_warmstart(policy, config, run, device):
    settings = config.get("behavior_cloning")
    if not settings:
        return None
    collect = reuse_teacher_sequences if settings.get("precollected_teacher") else (
        collect_teacher_sequences
    )
    dataset, summaries = collect(config.get("environment", {}), settings, run / "teacher-data")
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings.get("learning_rate", 1e-3)))
    logs = fit_sequences(policy, dataset, settings, optimizer, device, "expert",
                         run / "bc-progress.json")
    rounds = []
    for index, dagger in enumerate(settings.get("dagger_rounds", [])):
        round_settings = {**settings, **dagger}
        round_settings["data_seed"] = int(dagger.get("data_seed", settings.get("data_seed", 9101)))
        if "teacher_beta" not in dagger or "episodes" not in dagger:
            raise ValueError("Each DAgger round needs explicit teacher_beta and decision budgets")
        extra, evidence = collect_teacher_sequences(
            config.get("environment", {}), round_settings, run / f"dagger-data-{index}",
            student=policy, teacher_beta=float(dagger["teacher_beta"]),
        )
        dataset.extend(extra)
        phase = f"dagger-{index}"
        logs.extend(fit_sequences(policy, dataset, round_settings, optimizer, device, phase,
                                  run / "bc-progress.json"))
        rounds.append({"round": index, "dataset": evidence,
                       "teacher_beta": dagger["teacher_beta"],
                       "aggregate_samples": sum(len(ep["action"]) for ep in dataset)})
    summary = {"epochs": logs, "dataset": summaries, "dagger_rounds": rounds,
               "batching": settings.get("batching", "sequential_episodes"),
               "history": "continuous per-episode TBPTT; padded samples excluded from loss"}
    write_json(run / "behavior-cloning.json", summary)
    return summary
