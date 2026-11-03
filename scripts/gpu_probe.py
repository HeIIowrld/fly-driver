"""Hard gate for CUDA, real EGL cameras and recurrent policy on the allocated GPU."""

import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.policy import policy_from_config, tensor_observation
from scripts.run_campaign import campaign_for_index, merged

if not torch.cuda.is_available():
    raise RuntimeError("Allocated job cannot use CUDA")
device = torch.device("cuda:0")
config = json.loads(Path("input/train.json").read_text(encoding="utf-8"))
if "campaign" in config:
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        config = campaign_for_index(config, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    config = merged(config, config["campaign"]["arms"][0].get("overrides", {}))
environment = {**config.get("environment", {}), "mode": "V", "max_steps": 8}
env = FlyBrainEnv(**environment)
try:
    obs, _ = env.reset(seed=3001)
    from OpenGL import GL

    env.renderer._gl_context.make_current()
    gl_vendor = GL.glGetString(GL.GL_VENDOR).decode()
    gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
    if "nvidia" not in gl_vendor.lower():
        raise RuntimeError(f"Expected NVIDIA GPU camera renderer; got {gl_vendor}: {gl_renderer}")
    if obs["images"].std() < 1 or np.any(obs["body_valid"][-6:]):
        raise RuntimeError("Invalid camera/sensor-only observation")
    clipping = env.configuration()["render_clip_m"]
    if env.camera.near_m and not np.isclose(clipping["near"], env.camera.near_m):
        raise RuntimeError("Configured metric camera clipping was not applied")
    policy = policy_from_config(config).to(device)
    state = policy.initial_state()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    for _ in range(4):
        mean, value, state = policy(tensor_observation(obs, device), state.detach())
        loss = mean.square().mean() + value.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        obs, _, _, _, _ = env.step(torch.tanh(mean.detach())[0].cpu().numpy())
    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "model_device": str(next(policy.parameters()).device),
        "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "gl_vendor": gl_vendor,
        "gl_renderer": gl_renderer,
        "camera_std": float(obs["images"].std()),
        "oracle_valid_sum": 0,
        "loss": float(loss.detach()),
        "visual_schema": env.visual_schema,
        "actuation_mode": env.control.actuation_mode,
        "action_shape": list(mean.shape),
        "motion_input": bool(config.get("motion_input", False)),
        "render_clip_m": clipping,
        "camera_source_nonblack_fraction": float((env.camera_source_images.sum(-1) > 0).mean()),
        "ray_valid_fraction": float(env.sampler.valid.mean()),
    }
    (Path(os.environ.get("FLYBRAIN_JOB_OUTPUT", "output")) / "gpu-probe.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)
finally:
    env.close()
