"""Bounded EGL/D3D12 comparison of identical physics, optics and GPU PPO.

Run on the owned WSL ROCm runtime. Each case has a separate process group and
deadline, so a graphics/HIP stall cannot stop the running research campaign.
"""

import argparse
import faulthandler
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from scripts.run_campaign import merged


def trial_config(config):
    arm = config.get("campaign", {}).get("arms", [{}])[0]
    trial = merged(config, arm.get("overrides", {}))
    for key in ("campaign", "behavior_cloning", "local_screening", "validation"):
        trial.pop(key, None)
    trial.update(seed=5101, device="cuda:0", num_envs=4, torch_threads=1,
                 horizon=64, updates=3, epochs=2, checkpoint_interval=3)
    return trial


def graphics_environment(renderer):
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", LP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", MIOPEN_FIND_MODE="FAST")
    for key in ("GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME"):
        environment.pop(key, None)
    environment["MUJOCO_GL"] = "egl" if renderer == "egl" else "glfw"
    if renderer == "d3d12":
        environment.update(GALLIUM_DRIVER="d3d12", MESA_D3D12_DEFAULT_ADAPTER_NAME="AMD")
        environment["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/opt/rocm/lib"
    return environment


class Timings:
    def __init__(self):
        self.seconds = {}
        self.calls = {}

    def wrap(self, name, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started
                self.calls[name] = self.calls.get(name, 0) + 1
        return timed


class TimedPhysics:
    def __init__(self, module, timings):
        self.module = module
        self.mj_step = timings.wrap("physics_step", module.mj_step)
        self.mj_forward = timings.wrap("physics_forward", module.mj_forward)

    def __getattr__(self, name):
        return getattr(self.module, name)


def profile(config, run, ticks=120):
    from flybrain_drive.env import FlyBrainEnv

    env = FlyBrainEnv(**config["environment"])
    timings = Timings()
    env.mj = TimedPhysics(env.mj, timings)
    env._render_cameras = timings.wrap("camera_render_and_sample", env._render_cameras)
    env.motion_filters.update = timings.wrap("motion_filter", env.motion_filters.update)
    env.road.project = timings.wrap("road_projection", env.road.project)
    env.control.command = timings.wrap("intent_controller", env.control.command)
    traces, images = [], []
    try:
        started = time.perf_counter()
        observation, _ = env.reset(seed=5101, options={"start_x": 12.0})
        reset_seconds = time.perf_counter() - started
        from OpenGL import GL

        graphics = {key: GL.glGetString(value).decode() for key, value in (
            ("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION),
        )}
        timings.seconds.clear()
        timings.calls.clear()
        started = time.perf_counter()
        for tick in range(ticks):
            # Identical external commands in both cases: rendering cannot affect dynamics.
            action = np.array([0.0, 0.05 * np.sin(tick / 40), 0.0], np.float32)
            observation, _, terminated, truncated, info = env.step(action)
            traces.append(np.r_[env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()])
            images.append(observation["images"].copy())
            if terminated or truncated:
                break
        elapsed = time.perf_counter() - started
        np.savez_compressed(run / "fixed-command-trace.npz", physics=np.stack(traces),
                            images=np.stack(images))
        result = {
            "graphics": graphics, "reset_seconds": reset_seconds, "ticks": len(traces),
            "wall_seconds": elapsed, "transitions_per_second": len(traces) / elapsed,
            "seconds": timings.seconds, "calls": timings.calls, "last_reason": info["reason"],
            "timing_scope": "exclusive components inside env.step; reset/warm graphics excluded",
            "physics_sha256": hashlib.sha256(np.stack(traces).tobytes()).hexdigest(),
        }
        write_json(run / "profile.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        env.close()


def run_case(config, output):
    # Keep imports lazy: spawned CPU physics workers must not import Torch/ROCm.
    from flybrain_drive.parallel_learning import train_parallel

    faulthandler.dump_traceback_later(60, repeat=True)
    run = begin_run(output, config)
    profile(config, run)
    train_parallel(config, run / "ppo")
    training = json.loads((run / "ppo/training.json").read_text())
    warm = training["updates"][1:]
    seconds = sum(row["update_wall_seconds"] for row in warm)
    result = {
        "status": "completed", "warm_transitions_per_second":
        config["num_envs"] * config["horizon"] * len(warm) / seconds,
        "warm_environment_seconds": sum(row["environment_seconds"] for row in warm),
        "warm_policy_seconds": sum(row["policy_seconds"] for row in warm),
        "warm_optimizer_seconds": sum(row["optimizer_seconds"] for row in warm),
        "gpu_peak_allocated_bytes": max(row["gpu_peak_allocated_bytes"] for row in warm),
        "parameter_change_l2": training["parameter_change_l2"],
        "scope": "same physics/optics, 768 PPO transitions; throughput gate, no driving claim",
    }
    write_json(run / "result.json", result)
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(result), flush=True)


def compare_profiles(first, second):
    a, b = load_array(first), load_array(second)
    with a, b:
        same_shape = a["physics"].shape == b["physics"].shape
        return {
            "physics_equal": bool(same_shape and np.array_equal(a["physics"], b["physics"])),
            "physics_max_abs_difference":
            float(np.max(np.abs(a["physics"] - b["physics"]))) if same_shape else None,
            "rgb_mean_abs_difference":
            float(np.abs(a["images"].astype(float) - b["images"].astype(float)).mean())
            if a["images"].shape == b["images"].shape else None,
        }


def benchmark(config, output, timeout_seconds=180, order=("egl", "d3d12")):
    if os.name != "posix":
        raise RuntimeError("Run this comparison in the owned WSL Linux runtime")
    trial = trial_config(config)
    run = begin_run(output, trial)
    write_json(run / "input.json", trial)
    script = Path(__file__).resolve()
    (run / "benchmark-source.py").write_bytes(script.read_bytes())
    write_json(run / "script-sha256.json", {"script": str(script),
               "sha256": hashlib.sha256(script.read_bytes()).hexdigest()})
    rows = []
    for renderer in order:
        case = run / renderer
        command = [sys.executable, "-m", "scripts.benchmark_acceleration", "--case",
                   "--config", str(run / "input.json"), "--output", str(case)]
        with (run / (renderer + ".log")).open("w") as log:
            process = subprocess.Popen(command, env=graphics_environment(renderer),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(run / "live.json", {"renderer": renderer, "owned_process_group": process.pid,
                       "deadline_seconds": timeout_seconds, "phase": "running_case"})
            try:
                code = process.wait(timeout=timeout_seconds)
                row = {"renderer": renderer, "status": "completed" if code == 0 else "failed",
                       "returncode": code}
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                row = {"renderer": renderer, "status": "timeout",
                       "deadline_seconds": timeout_seconds, "owned_process_group": process.pid}
            if (case / "result.json").exists():
                row["ppo"] = json.loads((case / "result.json").read_text())
            if (case / "profile.json").exists():
                row["profile"] = json.loads((case / "profile.json").read_text())
            rows.append(row)
            write_json(run / "progress.json", rows)
    comparison = {}
    if all((run / renderer / "fixed-command-trace.npz").exists() for renderer in ("egl", "d3d12")):
        comparison = compare_profiles(run / "egl/fixed-command-trace.npz",
                                      run / "d3d12/fixed-command-trace.npz")
    result = {"cases": rows, "fixed_command_equivalence": comparison,
              "host_load": "existing local campaign remains active; shared-host diagnostic",
              "automatic_runtime_change": False}
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--order", nargs=2, choices=("egl", "d3d12"), default=["egl", "d3d12"])
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if len(set(args.order)) != 2:
        parser.error("--order must contain each renderer once")
    configuration = json.loads(args.config.read_text())
    if args.case:
        run_case(configuration, args.output)
    else:
        benchmark(configuration, args.output, args.timeout_seconds, args.order)
