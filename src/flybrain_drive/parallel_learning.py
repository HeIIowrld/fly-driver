"""Process-parallel MuJoCo rollouts and batched recurrent PPO on CPU/CUDA."""

import copy
import json
import time

import numpy as np
import torch

from .evidence import begin_run, write_json
from .graph import save_graph, sha256
from .learning import advantages, device_evidence, graph_for_config, save_checkpoint, select_device
from .policy import log_probability, policy_from_config
from .training_validation import TrainingValidator, validate_training_settings
from .vector_env import make_vector


def batch_observation(observation, device):
    return {key: torch.as_tensor(value, device=device) for key, value in observation.items()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def anchor_loss(means, reference, mask):
    if reference is None:
        return means.new_zeros(())
    return ((torch.tanh(means) - torch.tanh(reference)).square() * mask).sum() / (
        means.shape[0] * means.shape[1] * mask.sum()
    )


def make_optimizer(policy, config):
    critic_lr = config.get("critic_learning_rate")
    lr = float(config.get("learning_rate", 3e-4))
    if lr <= 0 or (critic_lr is not None and float(critic_lr) <= 0):
        raise ValueError("Optimizer learning rates must be positive")
    if critic_lr is None:
        return torch.optim.Adam(policy.parameters(), lr=lr)
    critic_ids = {id(p) for p in policy.critic.parameters()}
    return torch.optim.Adam([
        {"params": [p for p in policy.parameters() if id(p) not in critic_ids], "lr": lr},
        {"params": list(policy.critic.parameters()), "lr": float(critic_lr)},
    ])


def train_parallel(config, output):
    seed = int(config.get("seed", 3001))
    count = int(config.get("num_envs", 8))
    horizon = int(config.get("horizon", 256))
    updates = int(config.get("updates", 16))
    epochs = int(config.get("epochs", 4))
    checkpoint_interval = int(config.get("checkpoint_interval", 2))
    retain_interval = int(config.get("retain_checkpoint_interval", 0))
    anchor_coefficient = float(config.get("anchor_coefficient", 0))
    max_update_kl = config.get("max_update_kl")
    validate_training_settings(config)
    if retain_interval < 0 or anchor_coefficient < 0:
        raise ValueError("Checkpoint retention and anchor coefficient cannot be negative")
    if max_update_kl is not None and float(max_update_kl) <= 0:
        raise ValueError("max_update_kl must be positive")
    collector_restart = bool(config.get("collector_restart", True))
    target_kl = config.get("target_kl")
    if target_kl is not None and float(target_kl) <= 0:
        raise ValueError("target_kl must be positive")
    if count < 2 or min(horizon, updates, epochs, checkpoint_interval) < 1:
        raise ValueError("Invalid parallel PPO budget")
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    graph, graph_metadata = graph_for_config(config)
    run = begin_run(output, config)
    env = None
    started = time.perf_counter()
    try:
        device = select_device(config)
        env = make_vector(config)
        environments = env.call("configuration")
        write_json(
            run / "environment.json",
            {"workers": environments, "start_x_range": config.get("start_x_range")},
        )
        if graph is not None:
            save_graph(run / "graph.npz", graph, graph_metadata)
        policy = policy_from_config(config, graph).to(device)
        optimizer = make_optimizer(policy, config)
        write_json(
            run / "device.json",
            {
                **device_evidence(config, device, policy),
                "num_envs": count,
                "worker_pids": [worker.pid for worker in env.processes],
                "vectorization": "spawn processes + shared observation memory",
            },
        )
        if config.get("behavior_cloning"):
            from .imitation import teacher_warmstart

            save_checkpoint(run / "random-checkpoint.pt", policy, optimizer, config, graph, 0)
            teacher_warmstart(policy, config, run, device)
        save_checkpoint(run / "initial-checkpoint.pt", policy, optimizer, config, graph, 0)
        reference = copy.deepcopy(policy).requires_grad_(False) if anchor_coefficient else None
        anchor_mask = torch.as_tensor(
            config.get("anchor_mask", [1] * min(2, len(policy.action_mask))
                       + [0] * max(0, len(policy.action_mask) - 2)),
            dtype=torch.float32, device=device,
        )
        if anchor_mask.shape != policy.action_mask.shape or (anchor_mask < 0).any():
            raise ValueError("Anchor mask must match active action ports")
        anchor_mask = anchor_mask * policy.action_mask
        if reference is not None and float(anchor_mask.sum()) == 0:
            raise ValueError("Anchor must include at least one active action port")
        validator = TrainingValidator(config, run)
        reference_sha = sha256(run / "initial-checkpoint.pt") if reference is not None else None
        if validator.due(0):
            write_json(run / "live.json", {"phase": "development_validation", "update": 0})
            validator.evaluate(run / "initial-checkpoint.pt", 0)
        initial = torch.cat(
            [parameter.detach().flatten() for parameter in policy.parameters()]
        ).clone()
        logs = []
        obs, start_info = env.reset(seed=[seed + i for i in range(count)])
        state = policy.initial_state(count)
        reference_state = None if reference is None else reference.initial_state(count)
        reset_next = np.ones(count, dtype=bool)
        episode_steps = np.zeros(count, dtype=np.int64)
        episode_rewards = np.zeros(count)
        episode_start_progress = np.asarray(start_info["truth"]["progress_3d"]).copy()
        current_info = start_info
        ppo_started = time.perf_counter()
        startup_seconds = ppo_started - started
        initial_validation_seconds = validator.seconds
        stop_reason = "budget_complete"
        for update in range(updates):
            update_started = time.perf_counter()
            if collector_restart and update > 0:
                obs, current_info = env.reset(
                    seed=[seed + update * 100000 + i for i in range(count)]
                )
                state = policy.initial_state(count)
                reference_state = None if reference is None else reference.initial_state(count)
                reset_next = np.ones(count, dtype=bool)
                episode_steps[:] = 0
                episode_rewards[:] = 0
                episode_start_progress = np.asarray(current_info["truth"]["progress_3d"]).copy()
            rollout_state = state.detach().clone()
            start_position = [list(p) for p in current_info["truth"]["position_xyz"]]
            steps_at_start = episode_steps.copy()
            hidden_norm_at_start = state.norm(dim=1).detach().cpu().tolist()
            rows, latents, old_logps, values, rewards, next_values = [], [], [], [], [], []
            reference_means = []
            terminated_rows, done_rows, resets, episodes = [], [], [], []
            policy_seconds = environment_seconds = 0.0
            for tick in range(horizon):
                rows.append({key: value.copy() for key, value in obs.items()})
                resets.append(reset_next.copy())
                begin_policy = time.perf_counter()
                with torch.no_grad():
                    mean, value, state = policy(
                        batch_observation(obs, device),
                        state,
                        torch.tensor(reset_next, device=device),
                    )
                    latent = (
                        mean + policy.log_std.clamp(-5, 1).exp() * torch.randn_like(mean)
                    ) * policy.action_mask
                    logp = log_probability(mean, policy.log_std, latent, policy.action_mask)
                    actions = torch.tanh(latent).cpu().numpy()
                    if reference is not None:
                        frozen_mean, _, reference_state = reference(
                            batch_observation(obs, device), reference_state,
                            torch.tensor(reset_next, device=device),
                        )
                        reference_means.append(frozen_mean)
                policy_seconds += time.perf_counter() - begin_policy
                begin_env = time.perf_counter()
                next_obs, reward, terminated, truncated, info = env.step(actions)
                environment_seconds += time.perf_counter() - begin_env
                done = np.logical_or(terminated, truncated)
                episode_steps += 1
                episode_rewards += reward
                begin_policy = time.perf_counter()
                with torch.no_grad():
                    # Terminal observation is evaluated before any worker reset.
                    _, bootstrap, _ = policy(batch_observation(next_obs, device), state)
                # The required CPU transfer waits for the bootstrap result. Avoid
                # another device-wide fence at every 100 Hz control decision.
                bootstrap_cpu = bootstrap.cpu().numpy()
                policy_seconds += time.perf_counter() - begin_policy
                latents.append(latent.detach())
                old_logps.append(logp.cpu().numpy())
                values.append(value.cpu().numpy())
                next_values.append(bootstrap_cpu)
                rewards.append(reward.copy())
                terminated_rows.append(terminated.copy())
                done_rows.append(done.copy())
                if np.any(done):
                    for worker in np.flatnonzero(done):
                        episodes.append(
                            {
                                "tick": tick,
                                "worker": int(worker),
                                "reason": str(info["reason"][worker]),
                                "progress_3d": float(info["truth"]["progress_3d"][worker]),
                                "progress_delta": float(
                                    info["truth"]["progress_3d"][worker]
                                    - episode_start_progress[worker]
                                ),
                                "return": float(episode_rewards[worker]),
                                "steps": int(episode_steps[worker]),
                            }
                        )
                    next_obs, reset_info = env.reset(
                        seed=[
                            seed + update * 100000 + (tick + 1) * count + i if done[i] else None
                            for i in range(count)
                        ],
                        options={"reset_mask": done.copy()},
                    )
                    episode_steps[done] = 0
                    episode_rewards[done] = 0
                    episode_start_progress[done] = reset_info["truth"]["progress_3d"][done]
                    # AsyncVectorEnv returns reset info only for the reset rows.
                    for worker in np.flatnonzero(done):
                        info["truth"]["position_xyz"][worker] = reset_info["truth"]["position_xyz"][
                            worker
                        ]
                reset_next = done.copy()
                obs = next_obs
                current_info = info
                if tick % 32 == 0:
                    write_json(run / "live.json", {
                        "phase": "ppo_rollout", "update": update, "tick": tick,
                        "interactions": update * horizon * count + (tick + 1) * count,
                        "device": str(device),
                        "gpu_allocated_bytes": torch.cuda.memory_allocated(device)
                        if device.type == "cuda" else 0,
                        "gpu_reserved_bytes": torch.cuda.memory_reserved(device)
                        if device.type == "cuda" else 0,
                    })
            tensors = {
                key: torch.as_tensor(np.stack([row[key] for row in rows]), device=device)
                for key in rows[0]
            }
            old_values = np.stack(values)
            adv = advantages(
                np.stack(rewards),
                old_values,
                np.stack(next_values),
                np.stack(terminated_rows),
                np.stack(done_rows),
                config.get("gamma", 0.99),
                config.get("gae_lambda", 0.95),
            )
            returns = torch.as_tensor(adv + old_values, dtype=torch.float32, device=device)
            advantage = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=device)
            latent_tensor = torch.stack(latents)
            old_lp = torch.as_tensor(np.stack(old_logps), device=device)
            reset_tensor = torch.as_tensor(np.stack(resets), device=device)
            frozen_means = torch.stack(reference_means) if reference_means else None
            clip = config.get("clip", 0.2)
            synchronize(device)
            begin_optimizer = time.perf_counter()
            epochs_ran = 0
            kl_stopped = False
            rollback = None if max_update_kl is None else (
                copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())
            )
            for _ in range(epochs):
                means, predicted = policy.sequence_batch(tensors, reset_tensor, rollout_state)
                logp = log_probability(means, policy.log_std, latent_tensor, policy.action_mask)
                log_ratio = logp - old_lp
                ratio = torch.exp(log_ratio)
                approx_kl = ((ratio - 1) - log_ratio).mean()
                actor_loss = -torch.minimum(
                    ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
                ).mean()
                value_loss = 0.5 * (predicted - returns).square().mean()
                imitation_loss = anchor_loss(means, frozen_means, anchor_mask)
                loss = (actor_loss + float(config.get("value_coefficient", 0.5)) * value_loss
                        + anchor_coefficient * imitation_loss)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite batched PPO loss")
                if target_kl is not None and float(approx_kl.detach()) > 1.5 * float(target_kl):
                    kl_stopped = True
                    break
                optimizer.zero_grad()
                loss.backward()
                if not policy.critic_feature_grad:
                    critic_ids = {id(p) for p in policy.critic.parameters()}
                    actor_parameters = [p for p in policy.parameters() if id(p) not in critic_ids]
                    actor_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, 0.5)
                    critic_norm = torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), 0.5)
                    norm = torch.maximum(actor_norm, critic_norm)
                else:
                    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite batched PPO gradient")
                optimizer.step()
                epochs_ran += 1
            rejected_update = False
            attempted_kl = None
            if rollback is not None:
                with torch.no_grad():
                    attempted_mean, _ = policy.sequence_batch(tensors, reset_tensor, rollout_state)
                    attempted_logp = log_probability(
                        attempted_mean, policy.log_std, latent_tensor, policy.action_mask
                    )
                    attempted_ratio = attempted_logp - old_lp
                    attempted_kl = float((attempted_ratio.exp() - 1 - attempted_ratio).mean())
                if not np.isfinite(attempted_kl) or attempted_kl > float(max_update_kl):
                    policy.load_state_dict(rollback[0])
                    optimizer.load_state_dict(rollback[1])
                    rejected_update = True
            # Keep the numerical state across updates. Replay with updated weights from
            # the stored boundary state; gradients never cross that boundary (TBPTT).
            with torch.no_grad():
                final_means, final_values, state = policy.sequence_batch(
                    tensors, reset_tensor, rollout_state, return_state=True
                )
                final_logp = log_probability(
                    final_means, policy.log_std, latent_tensor, policy.action_mask
                )
                final_log_ratio = final_logp - old_lp
                final_ratio = torch.exp(final_log_ratio)
                final_kl = ((final_ratio - 1) - final_log_ratio).mean()
                clip_fraction = ((final_ratio - 1).abs() > clip).float().mean()
                return_variance = returns.var(unbiased=False)
                explained_variance = (
                    float(1 - (returns - final_values).var(unbiased=False) / return_variance)
                    if float(return_variance) > 1e-8
                    else None
                )
                latent_entropy = (
                    (torch.distributions.Normal(final_means, policy.log_std.clamp(-5, 1).exp())
                     .entropy() * policy.action_mask)
                    .sum(-1)
                    .mean()
                )
            synchronize(device)
            optimizer_seconds = time.perf_counter() - begin_optimizer
            interactions = (update + 1) * horizon * count
            elapsed = time.perf_counter() - started
            ppo_elapsed = (time.perf_counter() - ppo_started
                           - (validator.seconds - initial_validation_seconds))
            row = {
                "update": update,
                "interactions": interactions,
                "reward_sum": float(np.stack(rewards).sum()),
                "loss": float(loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "anchor_loss": float(imitation_loss.detach()),
                "anchor_coefficient": anchor_coefficient,
                "reference_checkpoint_sha256": reference_sha,
                "critic_feature_grad": policy.critic_feature_grad,
                "attempted_kl": attempted_kl,
                "update_rejected": rejected_update,
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std(unbiased=False)),
                "validation_seconds": validator.seconds,
                "episodes": episodes,
                "num_envs": count,
                "device": str(device),
                "start_position_xyz": start_position,
                "episode_steps_at_start": steps_at_start.tolist(),
                "episode_steps_at_end": episode_steps.tolist(),
                "hidden_norm_at_start": hidden_norm_at_start,
                "collector_end_reason": (
                    "explicit_collector_restart" if collector_restart else "continued_next_update"
                ),
                "recurrent_boundary": "stored detached state; updated-weight segment replay",
                "approx_kl": float(final_kl),
                "clip_fraction": float(clip_fraction),
                "explained_variance": explained_variance,
                "latent_gaussian_entropy": float(latent_entropy),
                "action_log_std": policy.log_std.detach().cpu().tolist(),
                "action_mask": policy.action_mask.detach().cpu().tolist(),
                "policy_parameters": sum(p.numel() for p in policy.parameters()),
                "epochs_ran": epochs_ran,
                "accepted_optimizer_steps": 0 if rejected_update else epochs_ran,
                "kl_early_stop": kl_stopped,
                "wall_seconds": elapsed,
                "transitions_per_second": interactions / elapsed,
                "startup_and_warmstart_wall_seconds": startup_seconds,
                "ppo_wall_seconds": ppo_elapsed,
                "ppo_transitions_per_second": interactions / ppo_elapsed,
                "policy_seconds": policy_seconds,
                "environment_seconds": environment_seconds,
                "optimizer_seconds": optimizer_seconds,
                "update_wall_seconds": time.perf_counter() - update_started,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else 0,
            }
            logs.append(row)
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            write_json(run / "progress.json", row)
            print(
                f"update={update} envs={count} interactions={interactions} "
                f"loss={float(loss.detach()):.5f} ppo_sps={interactions / ppo_elapsed:.1f}",
                flush=True,
            )
            if (update + 1) % checkpoint_interval == 0:
                save_checkpoint(run / "checkpoint.pt", policy, optimizer, config, graph, update + 1)
                write_json(run / "training.json", {"updates": logs})
            retained = retain_interval > 0 and (update + 1) % retain_interval == 0
            validation_due = validator.due(update + 1, final=update + 1 == updates)
            if retained or validation_due:
                snapshots = run / "checkpoints"
                snapshots.mkdir(exist_ok=True)
                snapshot = snapshots / f"update-{update + 1:06d}.pt"
                save_checkpoint(snapshot, policy, optimizer, config, graph, update + 1)
                if validation_due:
                    write_json(run / "live.json", {
                        "phase": "development_validation", "update": update + 1,
                        "interactions": interactions, "device": str(device),
                    })
                    if validator.evaluate(snapshot, update + 1):
                        stop_reason = "development_regression_stop"
                        break
        completed_updates = len(logs)
        save_checkpoint(run / "checkpoint.pt", policy, optimizer, config, graph, completed_updates)
        changed = (
            torch.cat([parameter.detach().flatten() for parameter in policy.parameters()]) - initial
        )
        write_json(
            run / "training.json",
            {
                "updates": logs,
                "interactions": count * horizon * completed_updates,
                "planned_updates": updates,
                "completed_updates": completed_updates,
                "stop_reason": stop_reason,
                "validation_seconds": validator.seconds,
                "num_envs": count,
                "collector_restart": collector_restart,
                "parameter_change_l2": float(torch.linalg.norm(changed)),
                "wall_seconds": time.perf_counter() - started,
                "startup_and_warmstart_wall_seconds": startup_seconds,
                "ppo_wall_seconds": (time.perf_counter() - ppo_started
                                     - (validator.seconds - initial_validation_seconds)),
            },
        )
        write_json(
            run / "result.json",
            {
                "status": "completed",
                "reason": stop_reason,
                "completed_updates": completed_updates,
                "checkpoint_sha256": sha256(run / "checkpoint.pt"),
            },
        )
        return run
    except Exception as error:
        write_json(
            run / "result.json",
            {"status": "failed", "reason": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        if env is not None:
            env.close(terminate=True)
