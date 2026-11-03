import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.imitation import fit_sequences
from flybrain_drive.learning import restore_policy
from flybrain_drive.policy import policy_from_config
from flybrain_drive.sensory_audit import evaluate_teacher
from flybrain_drive.sensory_imitation import collect_sequences, fixed_speed_action, train_sensory
from flybrain_drive.sensory_policy import SensoryFeatureEncoder
from flybrain_drive.sensory_runtime import make_sensory_runtime


def config():
    return {
        "seed": 17, "device": "cpu", "torch_threads": 1, "policy": "gru",
        "fixed_speed_mps": 1.5,
        "environment": {"mode": "V", "max_steps": 20, "vehicle": {"max_speed": 4},
                        "camera": {"fovy": 120, "source_size": 32}},
        "sensory_preprocessing": {"eye": {"rows": 2, "columns": 3, "sigma_degrees": 1}},
        "behavior_cloning": {"data_seed": 117, "epochs": 1, "chunk_length": 2,
                             "learning_rate": .001, "batching": "parallel_episodes",
                             "episodes": [{"start_x": 3, "steps": 4},
                                          {"start_x": 35, "steps": 3}],
                             "dagger_rounds": [{"data_seed": 217, "epochs": 1,
                                                "teacher_beta": 0,
                                                "episodes": [{"start_x": 3, "steps": 4}]}]},
        "evaluation": {"steps": 4, "cases": [{"name": "test", "seed": 991, "start_x": 3}]},
    }


def test_scalar_runtime_seals_calibration_and_does_not_invent_missing_telemetry():
    cfg = config()
    calibration = json.loads(Path("configs/vehicle-scalar-organs.json").read_text(encoding="utf8"))
    cfg["sensory_preprocessing"]["scalar_organs"] = calibration["scalar_organs"]
    env = FlyBrainEnv(**cfg["environment"])
    try:
        _, packer, effective = make_sensory_runtime(env, cfg)
        assert packer.organ_size == 132
        assert len(effective["sensory_encoder"]["organ_calibration_sha256"]) == 64
        assert "sensory_encoder" not in cfg
        changed = copy.deepcopy(effective)
        changed["sensory_preprocessing"]["scalar_organs"][0]["tuning_width"] += 1
        with pytest.raises(ValueError, match="calibration"):
            make_sensory_runtime(env, changed)
        # No new bridge reading is fabricated by opting into a new encoder.
        from flybrain_drive.sensor_contract import ActorObservation
        result = packer.pack(ActorObservation("missing-scalar-fixture", 0, 0, 0, (), ()))
        assert not result["organ_valid"][54:126].any()
        assert not result["organs"][54:126].any()
    finally:
        env.close()


def test_teacher_audit_records_pre_command_post_alignment(tmp_path):
    cfg = config()
    result = evaluate_teacher(cfg, cfg["evaluation"], tmp_path / "teacher")
    assert result["cases"][0]["steps"] == 4
    assert result["cases"][0]["reason"] == "time_limit"
    assert not result["cases"][0]["completion"]
    with np.load(tmp_path / "teacher/teacher-trace-00.npz") as arrays:
        trace = arrays["trace"]
    np.testing.assert_allclose(trace[:, 5] - trace[:, 0], .01)
    np.testing.assert_allclose(trace[:-1, 5:8], trace[1:, :3])
    np.testing.assert_allclose(trace[:, 3], -.25)
    evaluate_teacher(cfg, cfg["evaluation"], tmp_path / "teacher-no-render", render=False)
    with np.load(tmp_path / "teacher-no-render/teacher-trace-00.npz") as arrays:
        np.testing.assert_array_equal(trace, arrays["trace"])
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_teacher(cfg, {**cfg["evaluation"], "steps": 0}, tmp_path / "invalid")


def test_actual_sensor_dagger_records_labels_separately_and_enforces_fixed_speed(tmp_path):
    cfg = config()
    env = FlyBrainEnv(**cfg["environment"])
    try:
        _, _, sealed = make_sensory_runtime(env, cfg)
        assert "sensory_encoder" not in cfg
        student = policy_from_config(sealed)
        with torch.no_grad():
            student.actor.weight.zero_()
            student.actor.bias[:] = torch.tensor([.8, .7])
        command = fixed_speed_action([.99, .3], env, 1.5)
        np.testing.assert_allclose(command, [-.25, .3])
    finally:
        env.close()
    data, evidence = collect_sequences(sealed, cfg["behavior_cloning"], tmp_path / "dagger",
                                        student=student, teacher_beta=0)
    assert sum(len(d["action"]) for d in data) == 7
    assert all(row["teacher_decisions"] == 0 for row in evidence)
    with np.load(tmp_path / "dagger/episode-000.npz") as arrays:
        assert not arrays["teacher_selected"].any()
        assert np.max(np.abs(arrays["action"][:, 1] - arrays["executed_action"][:, 1])) > .1
        np.testing.assert_allclose(arrays["executed_action"][:, 0], -.25)
        actor_keys = {key[4:] for key in arrays.files if key.startswith("obs_")}
        assert actor_keys == SensoryFeatureEncoder.fields
        assert not any("truth" in key or "lateral_error" in key for key in actor_keys)


def test_supervision_mask_keeps_unsupervised_readout_row_unchanged(tmp_path):
    cfg = config()
    env = FlyBrainEnv(**cfg["environment"])
    try:
        _, _, sealed = make_sensory_runtime(env, cfg)
    finally:
        env.close()
    policy = policy_from_config(sealed)
    data, _ = collect_sequences(sealed, cfg["behavior_cloning"], tmp_path / "expert")
    before = policy.actor.weight.detach().clone()
    settings = {**cfg["behavior_cloning"], "supervised_action_mask": [0, 1]}
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    fit_sequences(policy, data, settings, optimizer, torch.device("cpu"), "steering")
    torch.testing.assert_close(policy.actor.weight[0], before[0], rtol=0, atol=0)
    assert not torch.equal(policy.actor.weight[1], before[1])
    with pytest.raises(ValueError, match="At least one"):
        fit_sequences(policy, data, {**settings, "supervised_action_mask": [0, 0]},
                      optimizer, torch.device("cpu"), "invalid")


@pytest.mark.parametrize("visual_encoder", ["global_points", "retinotopic_grid"])
def test_real_collection_training_dagger_evaluation_and_checkpoint_roundtrip(
    tmp_path, visual_encoder,
):
    cfg = config()
    cfg["sensory_preprocessing"]["visual_encoder"] = visual_encoder
    cfg["evaluate_vision_missing"] = True
    original = copy.deepcopy(cfg)
    run = train_sensory(cfg, tmp_path / "train")
    assert cfg == original
    summary = json.loads((run / "summary.json").read_text())
    assert summary["training_epochs"][0]["supervised_samples"] == 7
    assert summary["training_epochs"][1]["supervised_samples"] == 11
    assert summary["dagger_rounds"][0]["aggregate_samples"] == 11
    assert summary["vision_missing"]["aggregate"]["cases"] == 1
    restored, sealed = restore_policy(run / "checkpoint.pt")
    assert isinstance(restored.encoder, SensoryFeatureEncoder)
    assert sealed["behavior_cloning"]["supervised_action_mask"] == [0, 1]
    assert json.loads((run / "result.json").read_text())["goal_achieved"] is False
