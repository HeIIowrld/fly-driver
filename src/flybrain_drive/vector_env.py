"""Spawned MuJoCo workers, shared observations, explicit terminal-observation handling."""

import sys
from functools import partial
from pathlib import Path

import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from .env import FlyBrainEnv


class RandomStartEnv(FlyBrainEnv):
    def __init__(
        self, start_x_range=None, start_lateral_range=None, start_heading_range=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.start_x_range = start_x_range
        self.pose_ranges = {
            "lateral_offset": start_lateral_range,
            "heading_offset_degrees": start_heading_range,
        }
        for bounds in self.pose_ranges.values():
            if bounds is not None and (
                len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]
            ):
                raise ValueError("Invalid initial pose range")
        self.start_rng = np.random.default_rng()
        if start_x_range is not None and not (
            len(start_x_range) == 2
            and 0 <= start_x_range[0] < start_x_range[1] < self.road.config.length - 5
        ):
            raise ValueError("Invalid start_x_range")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.start_rng = np.random.default_rng(seed + 900000)
        options = dict(options or {})
        if self.start_x_range is not None and "start_x" not in options:
            options["start_x"] = float(self.start_rng.uniform(*self.start_x_range))
        for key, bounds in self.pose_ranges.items():
            if bounds is not None and key not in options:
                options[key] = float(self.start_rng.uniform(*bounds))
        return super().reset(seed=seed, options=options)

    def configuration(self):
        configuration = super().configuration()
        memory = {}
        path = Path("/proc/self/smaps_rollup")
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith(("Rss:", "Pss:")):
                    key, value, _ = line.split()
                    memory[key.rstrip(":").lower() + "_bytes"] = int(value) * 1024
        return {**configuration, "worker_runtime": {
            "torch_loaded": "torch" in sys.modules, **memory,
        }}


def make_vector(config):
    count = int(config.get("num_envs", 8))
    if count < 2:
        raise ValueError("Parallel trainer requires num_envs >=2")
    return AsyncVectorEnv(
        [
            partial(
                RandomStartEnv,
                start_x_range=config.get("start_x_range"),
                start_lateral_range=config.get("start_lateral_range"),
                start_heading_range=config.get("start_heading_range"),
                **config.get("environment", {}),
            )
            for _ in range(count)
        ],
        context="spawn",
        shared_memory=True,
        copy=True,
        autoreset_mode=AutoresetMode.DISABLED,
    )
