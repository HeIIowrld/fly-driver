import argparse
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("paramiko")

from scripts.remote_gpu import slurm_options
from scripts.run_campaign import campaign_for_index, merge_shards, run_campaign


def test_seed_arm_mapping_is_unique_and_does_not_modify_campaign():
    config = {"campaign": {"seeds": [3001, 3002], "arms": [{"name": name} for name in "abcd"]}}
    mapped = [campaign_for_index(config, index) for index in range(8)]
    pairs = [(c["campaign"]["seeds"][0], c["campaign"]["arms"][0]["name"]) for c in mapped]
    assert pairs == [(seed, name) for seed in (3001, 3002) for name in "abcd"]
    assert len(config["campaign"]["arms"]) == 4 and len(config["campaign"]["seeds"]) == 2
    for index in (-1, 8):
        with pytest.raises(ValueError, match="out of range"):
            campaign_for_index(config, index)


def test_array_has_one_gpu_at_a_time_and_merge_has_no_gpu_or_node_pin():
    args = argparse.Namespace(
        mode="campaign-array", after_job=None, partition="partition1", node="auto", time="00:30:00"
    )
    config = {"num_envs": 8, "campaign": {"seeds": [1, 2], "arms": [1, 2, 3, 4]}}
    options = slurm_options(args, config, "/owned-run", "ALL")
    assert "--array=0-7%1" in options and "--cpus-per-task=8" in options
    assert "--time=00:30:00" in options and "--gres=gpu:1" in options
    assert not any(o.startswith(("--nodelist=", "--dependency=")) for o in options)
    assert any("%A_%a.out" in o for o in options)
    merged = slurm_options(
        args, config, "/owned-run", "ALL", mode="campaign-merge", after_job="123"
    )
    assert "--cpus-per-task=1" in merged and "--dependency=afterany:123" in merged
    assert not any(o.startswith(("--array=", "--gres=")) for o in merged)
    with pytest.raises(ValueError, match="Invalid dependency"):
        slurm_options(args, config, "/owned-run", "ALL", after_job="123;bad")


def test_real_shard_training_and_merge_preserve_missing_task_failures(tmp_path):
    config = {
        "device": "cpu",
        "num_envs": 2,
        "horizon": 3,
        "updates": 1,
        "epochs": 1,
        "torch_threads": 1,
        "environment": {"mode": "O", "max_steps": 10},
        "campaign": {
            "seeds": [3001, 3002],
            "arms": [{"name": "a"}, {"name": "b"}],
            "evaluation": {"steps": 3, "cases": [{"name": "nominal"}]},
        },
    }
    shards = tmp_path / "shards"
    assert run_campaign(campaign_for_index(config, 2), shards / "task-2/campaign") == 0
    assert merge_shards(config, shards, tmp_path / "merged") == 3
    result = json.loads((tmp_path / "merged/result.json").read_text())
    assert result["status"] == "completed_with_failures"
    assert result["failed"] == 3 and result["training_interactions"] == 6
    summary = json.loads((tmp_path / "merged/summary.json").read_text())
    assert summary["runs"][2]["arm"] == "a" and summary["runs"][2]["seed"] == 3002
    assert summary["runs"][0]["reason"] == "missing_shard_summary"
    assert summary["paired_comparisons"] == []
