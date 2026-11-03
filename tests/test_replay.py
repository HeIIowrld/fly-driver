import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evaluation import evaluate_suite
from flybrain_drive.graph import sha256
from flybrain_drive.policy import policy_from_config


@pytest.mark.parametrize("retina", [False, True])
def test_recording_preserves_policy_trajectory_and_information_contract(tmp_path, retina):
    torch.manual_seed(12)
    config = {"environment": {"mode": "V"}, "torch_threads": 1}
    if retina:
        config.update(motion_input=True)
        config["environment"].update(
            visual_schema="retina_v1", actuation_mode="residual",
            camera={"sampling": "angular_hex", "source_size": 96, "fovy": 120},
        )
    policy = policy_from_config(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "graph": None,
            "config": config,
        },
        checkpoint,
    )
    suite = {"steps": 12, "cases": [{"name": "<camera-check>", "seed": 3}]}
    plain = evaluate_suite(checkpoint, tmp_path / "plain", suite)
    recorded = evaluate_suite(checkpoint, tmp_path / "recorded", suite, record=True)
    assert plain["aggregate"] == recorded["aggregate"]
    for name in ("trace-01.json",):
        assert (tmp_path / "plain" / name).read_bytes() == (
            tmp_path / "recorded" / name
        ).read_bytes()
    directory = tmp_path / "recorded" / "recording-01"
    metadata = json.loads((directory / "recording.json").read_text())
    for name, digest in metadata["files_sha256"].items():
        assert sha256(directory / name) == digest
    assert "&lt;camera-check&gt;" in (directory / "replay.html").read_text(encoding="utf-8")
    with np.load(directory / "samples.npz") as raw:
        assert raw["images"].shape == (4, 2, 32, 32, 3)
        assert raw["observer"].shape == (4, 128, 128, 3)
        assert raw["images"].std() > 1 and raw["observer"].std() > 1
        if retina:
            assert raw["camera_source"].shape == (4, 2, 96, 96, 3)
            assert raw["action"].shape == (4, 3)
            assert not raw["motion_valid"][0].any()
            assert raw["motion_valid"][1].any()
        np.testing.assert_array_equal(raw["body"][:, -6:], 0)
        np.testing.assert_array_equal(raw["body_valid"][:, -6:], 0)
        np.testing.assert_array_equal(raw["action_valid"], [True, True, True, False])
        np.testing.assert_array_equal(raw["tick"], [0, 4, 8, 12])
        trace = json.loads((tmp_path / "recorded" / "trace-01.json").read_text())
        for index, tick in enumerate(raw["tick"][:-1]):
            np.testing.assert_array_equal(
                raw["applied_control"][index], trace[tick]["applied_actuator_control"]
            )
            assert raw["time"][index] == trace[tick]["observation_time"]


def test_camera_exposure_clock_is_25_hz_for_requested_30_at_100_hz_control():
    env = FlyBrainEnv(mode="V", camera_hz=30)
    try:
        _, info = env.reset(seed=2)
        exposures = [info["time"]]
        for _ in range(12):
            obs, _, _, _, info = env.step(np.array([-0.75, 0], np.float32))
            if obs["camera_age"][0] < 1e-8:
                exposures.append(info["time"])
        np.testing.assert_allclose(np.diff(exposures), 0.04, atol=1e-8)
    finally:
        env.close()
