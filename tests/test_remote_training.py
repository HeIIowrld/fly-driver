import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from flybrain_drive.learning import evaluate, select_device, train


def test_explicit_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        select_device({"device": "cuda:0"})
    assert select_device({"device": "cpu"}).type == "cpu"


def test_periodic_checkpoint_and_random_terrain_start(tmp_path):
    config = {
        "policy": "gru",
        "seed": 3001,
        "device": "cpu",
        "start_x_range": [20, 60],
        "horizon": 8,
        "updates": 2,
        "epochs": 1,
        "checkpoint_interval": 1,
        "environment": {"mode": "O", "max_steps": 8},
    }
    train(config, tmp_path / "training")
    training = json.loads((tmp_path / "training/training.json").read_text())
    assert training["interactions"] == 16
    assert training["parameter_change_l2"] > 0
    starts = [row["start_position_xyz"][0] for row in training["updates"]]
    assert all(19 < x < 61 for x in starts) and starts[0] != starts[1]
    checkpoint = torch.load(tmp_path / "training/checkpoint.pt", weights_only=True)
    assert checkpoint["completed_updates"] == 2
    assert not list((tmp_path / "training").glob("*.partial"))
    evaluate(tmp_path / "training/checkpoint.pt", tmp_path / "evaluation", steps=4)
    assert json.loads((tmp_path / "evaluation/result.json").read_text())["status"] == "completed"
