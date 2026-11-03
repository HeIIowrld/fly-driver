"""Preserve experimental inputs and verify renderer-only equivalence reporting."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.benchmark_acceleration import compare_profiles, graphics_environment, trial_config


def test_acceleration_trial_keeps_physics_optics_and_actor_contract():
    root = Path(__file__).resolve().parents[1]
    configuration = json.loads((root / "configs/campaign-rdna4-screening.json").read_text())
    original = json.dumps(configuration, sort_keys=True)
    trial = trial_config(configuration)
    assert json.dumps(configuration, sort_keys=True) == original
    assert trial["environment"]["physics_dt"] == 0.001
    assert trial["environment"]["control_dt"] == 0.01
    assert trial["environment"]["mode"] == "V"
    assert trial["environment"]["camera_hz"] == 25
    assert trial["environment"]["camera"]["source_size"] == 96
    assert trial["environment"]["camera"]["sampling"] == "angular_hex"
    assert trial["environment"]["camera"]["near_m"] == 0.03
    assert trial["environment"]["actuation_mode"] == "residual"
    assert trial["motion_input"]
    assert "behavior_cloning" not in trial and "campaign" not in trial
    assert trial["num_envs"] * trial["horizon"] * trial["updates"] == 768


def test_renderer_environment_does_not_leak_previous_driver(monkeypatch):
    monkeypatch.setenv("GALLIUM_DRIVER", "other")
    monkeypatch.setenv("MESA_D3D12_DEFAULT_ADAPTER_NAME", "other")
    software = graphics_environment("egl")
    hardware = graphics_environment("d3d12")
    assert "GALLIUM_DRIVER" not in software
    assert "MESA_D3D12_DEFAULT_ADAPTER_NAME" not in software
    assert software["MUJOCO_GL"] == "egl"
    assert hardware["GALLIUM_DRIVER"] == "d3d12"
    assert hardware["MESA_D3D12_DEFAULT_ADAPTER_NAME"] == "AMD"
    assert hardware["MUJOCO_GL"] == "glfw"
    assert hardware["MIOPEN_FIND_MODE"] == software["MIOPEN_FIND_MODE"] == "FAST"


def test_equivalence_distinguishes_physics_from_renderer_pixels(tmp_path):
    state = np.arange(12, dtype=float).reshape(3, 4)
    image = np.zeros((3, 2, 16, 16, 3), np.uint8)
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    np.savez(first, physics=state, images=image)
    np.savez(second, physics=state, images=image + 2)
    result = compare_profiles(first, second)
    assert result["physics_equal"] and result["physics_max_abs_difference"] == 0
    assert result["rgb_mean_abs_difference"] == 2
    np.savez(second, physics=state + 0.1, images=image)
    assert not compare_profiles(first, second)["physics_equal"]


def test_benchmark_spawn_entrypoint_keeps_torch_out_of_cpu_workers():
    result = subprocess.check_output([
        sys.executable, "-c", "import sys; import scripts.benchmark_acceleration; "
        "print('torch' in sys.modules)",
    ], text=True)
    assert result.strip() == "False"
