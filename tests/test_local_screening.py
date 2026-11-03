import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.imitation import collect_teacher_sequences, reuse_teacher_sequences
from flybrain_drive.learning import device_evidence
from flybrain_drive.policy import policy_from_config
from scripts.select_local_candidates import select_candidates
from scripts.watch_local_screening import collect_tree


def test_spawn_entrypoints_do_not_import_learner_into_cpu_workers():
    for name in ("scripts.run_campaign", "scripts.benchmark_learning"):
        subprocess.run([sys.executable, "-c", f"import {name}; import sys; "
                        "assert 'torch' not in sys.modules"], check=True)


def test_efference_ablation_removes_commands_without_changing_physics():
    config = {"mode": "V", "visual_schema": "retina_v1", "actuation_mode": "residual"}
    full = FlyBrainEnv(**config)
    ablated = FlyBrainEnv(**config, sensor_profile="no_efference_copy")
    try:
        full.reset(seed=3101)
        ablated.reset(seed=3101)
        action = np.array([-.25, .2, .5], np.float32)
        a, *_ = full.step(action)
        b, *_ = ablated.step(action)
        np.testing.assert_array_equal(full.data.qpos, ablated.data.qpos)
        np.testing.assert_array_equal(full.data.qvel, ablated.data.qvel)
        np.testing.assert_array_equal(full.data.ctrl, ablated.data.ctrl)
        assert a["body_valid"][20:24].all() and np.any(a["body"][20:24])
        assert not b["body_valid"][20:24].any() and not b["body"][20:24].any()
        assert a["previous_yaw_action"][0] == .5
        assert b["previous_yaw_action"][0] == 0
        for key in ("images", "motion", "body", "body_valid"):
            if key.startswith("body"):
                np.testing.assert_array_equal(a[key][:20], b[key][:20])
                np.testing.assert_array_equal(a[key][24:], b[key][24:])
            else:
                np.testing.assert_array_equal(a[key], b[key])
    finally:
        full.close()
        ablated.close()


def test_cpu_evidence_does_not_claim_gpu_execution():
    evidence = device_evidence({}, torch.device("cpu"), policy_from_config({}))
    assert evidence["backend"] == "cpu"
    assert evidence["model_device"] == "cpu"
    assert evidence["gpu"] is None and evidence["gpu_total_memory_bytes"] is None


def test_promotion_needs_paired_replication_and_does_not_hide_failed_runs():
    def row(arm, seed, progress, failures):
        return {"arm": arm, "seed": seed, "status": "completed", "final": {
            "cases": [{"case": str(i), "progress_delta": progress, "failure": i < failures}
                      for i in range(4)],
            "aggregate": {"mean_progress_delta": progress},
        }}

    runs = [row("full_mixed", 4101, 5, 4), row("candidate", 4101, 7, 4)]
    selection = select_candidates({"runs": runs})
    assert selection["replication_candidates"] == ["candidate"]
    assert selection["remote_addition_candidates"] == []
    runs.extend([row("full_mixed", 4102, 6, 4),
                 {"arm": "candidate", "seed": 4102, "status": "failed"}])
    assert select_candidates({"runs": runs})["replication_candidates"] == []
    runs[-1] = row("candidate", 4102, 8, 4)
    assert select_candidates({"runs": runs})["remote_addition_candidates"] == ["candidate"]


def test_screening_followup_keeps_long_remote_budget_and_existing_suite(tmp_path, monkeypatch):
    from scripts import finish_local_screening

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    config = json.loads((root / "configs/campaign-rdna4-screening.json").read_text())
    config["behavior_cloning"]["precollected_teacher"] = "/only/local/expert"

    def rows(seed):
        return [{"arm": name, "seed": seed, "status": "completed", "final": {
            "cases": [{"case": str(i), "progress_delta": distance, "failure": True}
                      for i in range(4)],
            "aggregate": {"mean_progress_delta": distance},
        }} for name, distance in (("full_mixed", 5), ("no_copy_student", 7))]

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "result.json").write_text('{"status":"completed"}')
    (stage / "summary.json").write_text(json.dumps({"runs": rows(4101)}))

    def fake_replica(replica, output):
        assert replica["updates"] == 16 and replica["campaign"]["seeds"] == [4102]
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"runs": rows(4102)}))
        return 0

    monkeypatch.setattr(finish_local_screening, "run_campaign", fake_replica)
    output = tmp_path / "pipeline"
    finish_local_screening.finish(config, stage, output, 999999)
    remote = json.loads((output / "remote-addition-config.json").read_text())
    assert remote["updates"] == 32 and remote["num_envs"] == 8
    assert remote["campaign"]["seeds"] == [4101, 4102]
    assert [a["name"] for a in remote["campaign"]["arms"]] == ["full_mixed", "no_copy_student"]
    assert remote["campaign"]["evaluation"]["cases"][0]["seed"] == 8001
    assert "/only/local/expert" not in json.dumps(remote)
    assert config["updates"] == 16 and len(config["campaign"]["arms"]) == 4


def test_completed_local_collection_verifies_and_preserves_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "collected"
    source.mkdir()
    (source / "checkpoint.pt").write_bytes(b'checkpoint')
    (source / "active.partial").write_bytes(b'partial')
    receipt = collect_tree(source, target)
    assert list(receipt) == ["checkpoint.pt"]
    assert (target / "checkpoint.pt").read_bytes() == (source / "checkpoint.pt").read_bytes()
    assert not (target / "active.partial").exists()


def test_verified_expert_reuse_masks_history_and_rejects_changed_camera_or_data(tmp_path):
    environment = {"mode": "V", "visual_schema": "retina_v1", "actuation_mode": "residual"}
    settings = {"data_seed": 7101, "target_speed": 2, "teacher_actuation_mode": "fixed",
                "episodes": [{"start_x": 3, "steps": 4}]}
    original, _ = collect_teacher_sequences(environment, settings, tmp_path / "expert")
    reuse = {**settings, "precollected_teacher": str(tmp_path / "expert")}
    derived, _ = reuse_teacher_sequences(
        {**environment, "sensor_profile": "no_efference_copy"}, reuse, tmp_path / "derived"
    )
    np.testing.assert_array_equal(original[0]["action"], derived[0]["action"])
    np.testing.assert_array_equal(original[0]["observation"]["images"],
                                  derived[0]["observation"]["images"])
    assert not derived[0]["observation"]["body_valid"][:, 20:24].any()
    assert not derived[0]["observation"]["body"][:, 20:24].any()
    with pytest.raises(ValueError, match="incompatible"):
        reuse_teacher_sequences({**environment, "camera": {"forward": .95}},
                                reuse, tmp_path / "invalid-camera")
    path = tmp_path / "expert/episode-00.npz"
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match="hash mismatch"):
        reuse_teacher_sequences(environment, reuse, tmp_path / "invalid-data")
