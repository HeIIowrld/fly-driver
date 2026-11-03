"""Validate actual CNN/recurrent GPU training and identify the camera renderer."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.learning import device_evidence, select_device
from flybrain_drive.policy import policy_from_config, tensor_observation


def probe(config, output):
    run = begin_run(output, config)
    device = select_device(config)
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    torch.manual_seed(3101)
    env = FlyBrainEnv(**config["environment"])
    try:
        observation, _ = env.reset(seed=3101)
        from OpenGL import GL

        renderer = {
            "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
            "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
            "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
            "version": GL.glGetString(GL.GL_VERSION).decode(),
        }
        policy = policy_from_config(config).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0005)
        obs = tensor_observation(observation, device)
        obs = {key: value.repeat(8, *([1] * (value.ndim - 1))) for key, value in obs.items()}
        initial = policy.actor.weight.detach().clone()
        times, losses = [], []
        for _iteration in range(8):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            state = policy.initial_state(8)
            optimizer.zero_grad()
            loss = torch.zeros((), device=device)
            for _ in range(16):
                mean, value, state = policy(obs, state)
                loss = loss + (mean - 0.1).square().mean() + value.square().mean()
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters()):
                raise RuntimeError("Non-finite GPU gradients")
            optimizer.step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - started)
            losses.append(float(loss.detach()))
        change = float((policy.actor.weight.detach() - initial).norm())
        if change <= 0 or losses[-1] >= losses[0]:
            raise RuntimeError("GPU optimizer did not change parameters/reduce diagnostic loss")
        array = np.arange(4, dtype=np.float32)
        if not np.array_equal(torch.from_numpy(array).to(device).cpu().numpy(), array):
            raise RuntimeError("NumPy GPU round trip failed")
        result = {
            "status": "passed",
            "device": device_evidence(config, device, policy),
            "renderer": renderer,
            "test": "8 optimizer steps, 16 recurrent ticks, batch 8, CNN+GRU backward",
            "losses": losses,
            "actor_parameter_change_l2": change,
            "iteration_seconds": times,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "physics": "CPU MuJoCo, GPU actor/optimizer; renderer listed separately",
        }
        write_json(run / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(run / "result.json", {
            "status": "failed", "reason": type(error).__name__, "message": str(error),
        })
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(json.loads(args.config.read_text()), args.output)
