"""Single-environment full-sequence recurrent PPO prototype, with timeout bootstrap."""

import time
from pathlib import Path

import numpy as np
import torch

from .env import FlyBrainEnv
from .evidence import begin_run, write_json
from .graph import demo_graph, load_graph, rewire, save_graph, sha256
from .policy import log_probability, policy_from_config, tensor_observation


def select_device(config):
    device = torch.device(config.get("device", "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA/ROCm) was requested but is unavailable; no CPU fallback")
    return device


def device_evidence(config, device, policy):
    """ROCm shares torch.cuda's API; record the actual accelerator separately."""
    gpu = device.type == "cuda"
    properties = torch.cuda.get_device_properties(device) if gpu else None
    return {
        "requested": config.get("device", "cpu"),
        "actual": str(device),
        "model_device": str(next(policy.parameters()).device),
        "backend": "rocm" if gpu and torch.version.hip else "cuda" if gpu else "cpu",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "gpu": properties.name if properties else None,
        "gpu_total_memory_bytes": properties.total_memory if properties else None,
        "gpu_architecture": getattr(properties, "gcnArchName", None),
    }


def save_checkpoint(path, policy, optimizer, config, graph, completed_updates):
    serial_graph = {key: value.tolist() for key, value in graph.items()} if graph else None
    partial = Path(path).with_suffix(".partial")
    torch.save(
        {
            "config": config,
            "graph": serial_graph,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if policy.log_std.is_cuda else [],
            "numpy_rng": np.random.get_state()[1].tolist(),
            "completed_updates": completed_updates,
            "resume_support": "not implemented; checkpoint is evaluation-ready",
        },
        partial,
    )
    partial.replace(path)


def graph_for_config(config):
    if config.get("policy", "gru") == "gru":
        return None, None
    path = config.get("graph_path")
    if path:
        graph = load_graph(path)
        metadata = {"source": "user_reviewed_graph", "source_sha256": sha256(path)}
    elif config.get("graph_source") == "synthetic_demo":
        graph = demo_graph(config.get("graph_seed", 7))
        metadata = {"source": "synthetic_demo_not_fly", "graph_seed": config.get("graph_seed", 7)}
    else:
        raise ValueError("Graph policy needs reviewed graph_path or explicit synthetic_demo")
    if config.get("rewire_seed") is not None:
        graph, rewiring = rewire(graph, config["rewire_seed"])
        metadata["rewiring"] = rewiring
    return graph, metadata


def advantages(rewards, values, next_values, terminated, done, gamma=0.99, lam=0.95):
    """Timeouts bootstrap a terminal observation but never continue into the reset episode."""
    result = np.zeros_like(rewards, dtype=np.float32)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - terminated[i]) - values[i]
        carry = delta + gamma * lam * (1 - done[i]) * carry
        result[i] = carry
    return result


def train(config, output):
    seed = int(config.get("seed", 1))
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    horizon, updates = int(config.get("horizon", 128)), int(config.get("updates", 2))
    epochs = int(config.get("epochs", 2))
    if min(horizon, updates, epochs) < 1:
        raise ValueError("horizon, updates and epochs must be positive")
    if int(config.get("checkpoint_interval", 10)) < 1:
        raise ValueError("checkpoint_interval must be positive")
    graph, graph_metadata = graph_for_config(config)
    config = {
        **config,
        "graph_sha256": sha256(config["graph_path"]) if config.get("graph_path") else None,
    }
    run = begin_run(output, config)
    env = None
    started = time.perf_counter()
    try:
        device = select_device(config)
        env = FlyBrainEnv(**config.get("environment", {}))
        write_json(run / "environment.json", env.configuration())
        if graph is not None:
            save_graph(run / "graph.npz", graph, graph_metadata)
        policy = policy_from_config(config, graph).to(device)
        write_json(
            run / "device.json",
            device_evidence(config, device, policy),
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.get("learning_rate", 3e-4))
        if config.get("behavior_cloning"):
            from .imitation import teacher_warmstart

            save_checkpoint(run / "random-checkpoint.pt", policy, optimizer, config, graph, 0)
            teacher_warmstart(policy, config, run, device)
        save_checkpoint(run / "initial-checkpoint.pt", policy, optimizer, config, graph, 0)
        initial = torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()
        logs = []
        start_range = config.get("start_x_range")
        if start_range is not None:
            if (
                len(start_range) != 2
                or not 0 <= start_range[0] < start_range[1] < env.road.config.length - 5
            ):
                raise ValueError("Invalid training start_x_range")

        def reset_training(reset_seed):
            options = {"start_x": float(np.random.uniform(*start_range))} if start_range else None
            return env.reset(seed=reset_seed, options=options)

        for update in range(updates):
            # Explicit collector restart: each optimized sequence starts at reset.
            # The final value bootstraps the collector horizon; no stale hidden state reuse.
            obs, start_info = reset_training(seed + update * 10000)
            state = policy.initial_state()
            rows, latent_rows, old_logp, values, rewards, next_values = [], [], [], [], [], []
            terminated_rows, done_rows, resets = [], [], []
            reset_next = True
            episodes = []
            for tick in range(horizon):
                rows.append({key: value.copy() for key, value in obs.items()})
                resets.append(reset_next)
                with torch.no_grad():
                    mean, value, state = policy(
                        tensor_observation(obs, device),
                        state,
                        torch.tensor([reset_next], device=device),
                    )
                    latent = (
                        mean + policy.log_std.clamp(-5, 1).exp() * torch.randn_like(mean)
                    ) * policy.action_mask
                    lp = log_probability(mean, policy.log_std, latent, policy.action_mask)
                next_obs, reward, terminated, truncated, info = env.step(
                    torch.tanh(latent)[0].cpu().numpy()
                )
                done = terminated or truncated
                with torch.no_grad():
                    _, next_value, _ = policy(tensor_observation(next_obs, device), state)
                latent_rows.append(latent[0])
                old_logp.append(float(lp[0]))
                values.append(float(value[0]))
                next_values.append(float(next_value[0]))
                rewards.append(reward)
                terminated_rows.append(terminated)
                done_rows.append(done)
                reset_next = done
                if done:
                    episodes.append(
                        {
                            "tick": tick,
                            "reason": info["reason"],
                            "progress_3d": info["truth"]["progress_3d"],
                        }
                    )
                    next_obs, _ = reset_training(seed + update * 10000 + tick + 1)
                obs = next_obs
            tensors = {
                key: torch.as_tensor(np.stack([row[key] for row in rows]), device=device)
                for key in rows[0]
            }
            adv = advantages(
                np.array(rewards),
                np.array(values),
                np.array(next_values),
                np.array(terminated_rows),
                np.array(done_rows),
                config.get("gamma", 0.99),
                config.get("gae_lambda", 0.95),
            )
            returns = torch.tensor(adv + np.array(values), dtype=torch.float32, device=device)
            advantage = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=device)
            old_lp = torch.tensor(old_logp, device=device)
            latent_tensor = torch.stack(latent_rows)
            reset_tensor = torch.tensor(resets, device=device)
            clip = config.get("clip", 0.2)
            for _ in range(epochs):
                mean, predicted = policy.sequence(tensors, reset_tensor)
                lp = log_probability(mean, policy.log_std, latent_tensor, policy.action_mask)
                ratio = torch.exp(lp - old_lp)
                actor_loss = -torch.minimum(
                    ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
                ).mean()
                value_loss = 0.5 * (predicted - returns).square().mean()
                loss = actor_loss + float(config.get("value_coefficient", 0.5)) * value_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite PPO loss")
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Nonfinite PPO gradient")
                optimizer.step()
            logs.append(
                {
                    "update": update,
                    "reward_sum": float(sum(rewards)),
                    "loss": float(loss.detach()),
                    "episodes": episodes,
                    "collector_end_reason": "collector_horizon",
                    "interactions": (update + 1) * horizon,
                    "start_position_xyz": start_info["truth"]["position_xyz"],
                }
            )
            print(
                f"update={update} interactions={(update + 1) * horizon} "
                f"loss={float(loss.detach()):.5f}",
                flush=True,
            )
            write_json(
                run / "progress.json",
                {
                    "update": update,
                    "interactions": (update + 1) * horizon,
                    "loss": float(loss.detach()),
                    "device": str(device),
                    "wall_seconds": time.perf_counter() - started,
                    "last_episodes": episodes,
                },
            )
            if (update + 1) % int(config.get("checkpoint_interval", 10)) == 0:
                save_checkpoint(run / "checkpoint.pt", policy, optimizer, config, graph, update + 1)
                write_json(run / "training.json", {"updates": logs})
        changed = torch.cat([p.detach().flatten() for p in policy.parameters()]) - initial
        # Graph tensors are bundled in state_dict; explicit graph definition restores dimensions.
        save_checkpoint(run / "checkpoint.pt", policy, optimizer, config, graph, updates)
        write_json(
            run / "training.json",
            {
                "updates": logs,
                "interactions": horizon * updates,
                "parameter_change_l2": float(torch.linalg.norm(changed)),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        write_json(
            run / "result.json",
            {
                "status": "completed",
                "reason": "budget_complete",
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
            env.close()


def restore_policy(checkpoint):
    """Canonical CPU checkpoint reader for single and multi-case evaluation."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    graph = (
        {
            key: np.array(
                value,
                dtype=np.int64
                if key in ("pre", "post")
                else np.float32
                if key == "weights"
                else str,
            )
            for key, value in payload["graph"].items()
        }
        if payload["graph"]
        else None
    )
    policy = policy_from_config(config, graph)
    policy.load_state_dict(payload["model"])
    policy.eval()
    return policy, config


def evaluate(checkpoint, output, steps=300):
    if steps < 1:
        raise ValueError("steps must be positive")
    policy, config = restore_policy(checkpoint)
    run = begin_run(
        output, {"checkpoint_sha256": sha256(checkpoint), "steps": steps, "training_config": config}
    )
    env = None
    try:
        env = FlyBrainEnv(**config.get("environment", {}))
        write_json(run / "environment.json", env.configuration())
        state = policy.initial_state()
        obs, info = env.reset(seed=int(config.get("seed", 1)) + 100000)
        reward_sum = 0.0
        trace = []
        for _ in range(steps):
            with torch.no_grad():
                mean, _, state = policy(tensor_observation(obs), state)
            action = torch.tanh(mean)[0].numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            trace.append({"action": action.tolist(), **info})
            if terminated or truncated:
                break
        write_json(
            run / "evaluation.json",
            {
                "steps": len(trace),
                "reward_sum": reward_sum,
                "reason": info["reason"] if not env.active else "evaluation_budget",
                "final": info,
            },
        )
        write_json(run / "trace.json", trace)
        write_json(run / "result.json", {"status": "completed"})
        return run
    except Exception as error:
        write_json(
            run / "result.json",
            {"status": "failed", "reason": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        if env is not None:
            env.close()
