import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from flybrain_drive.graph import sha256
from scripts.prepare_remote_main import prepare_main
from scripts.select_local_candidates import select_candidates


def proposal():
    def row(name, seed, distance):
        return {"arm": name, "seed": seed, "status": "completed", "final": {
            "cases": [{"case": str(i), "progress_delta": distance, "failure": True}
                      for i in range(4)], "aggregate": {"mean_progress_delta": distance},
        }}

    runs = [row(name, seed, distance) for seed, values in (
        (4101, {"full_mixed": 5, "a": 7.5, "b": 8, "c": 7}),
        (4102, {"full_mixed": 6, "a": 9, "b": 7.8, "c": 9}),
    ) for name, distance in values.items()]
    return {
        "num_envs": 8, "horizon": 256, "updates": 32,
        "environment": {"mode": "V", "road": {"grade": .02}},
        "behavior_cloning": {"precollected_teacher": "/only/local/expert", "epochs": 24},
        "campaign": {"seeds": [4101, 4102], "arms": [
            {"name": name, "overrides": {"motion_input": True}}
            for name in ("full_mixed", "a", "b", "c")
        ], "evaluation": {"steps": 5000, "cases": [{"name": "held", "seed": 8001}]}},
        "local_selection": select_candidates({"runs": runs}),
    }


def main_policy():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/local-to-remote-main.json").read_text())


def test_main_uses_robust_top_two_fresh_seeds_and_preserves_sensor_evaluation_contract():
    original = proposal()
    untouched = copy.deepcopy(original)
    main = prepare_main(original, main_policy())
    assert original == untouched
    assert main["main_training"]["selected"] == ["a", "c"]
    assert [arm["name"] for arm in main["campaign"]["arms"]] == ["full_mixed", "a", "c"]
    assert main["campaign"]["seeds"] == [6101, 6102, 6103]
    assert main["main_training"]["transitions_per_task"] == 262144
    assert main["main_training"]["task_count"] == 9
    assert main["environment"] == original["environment"]
    assert main["campaign"]["evaluation"] == original["campaign"]["evaluation"]
    assert "/only/local" not in json.dumps(main)


def test_main_rejects_seed_leakage_and_failed_or_missing_replication():
    config, policy = proposal(), main_policy()
    policy["training"]["seeds"] = [4101, 6102, 6103]
    with pytest.raises(ValueError, match="fresh seeds"):
        prepare_main(config, policy)
    config["local_selection"]["evidence"][0]["paired_seeds"][0]["passed"] = False
    with pytest.raises(ValueError, match="replication"):
        prepare_main(config, main_policy())
    config = proposal()
    config["local_selection"]["remote_addition_candidates"] = []
    with pytest.raises(ValueError, match="eligible"):
        prepare_main(config, main_policy())


def test_remote_concurrency_is_configurable_bounded_and_cpu_merge_is_independent():
    pytest.importorskip("paramiko")
    from scripts.remote_gpu import slurm_options

    args = argparse.Namespace(mode="campaign-array", after_job=None, partition="partition1",
                              node="auto", time="03:00:00", array_concurrency=2)
    main = prepare_main(proposal(), main_policy())
    options = slurm_options(args, main, "/owned", "ALL")
    assert "--array=0-8%2" in options and "--gres=gpu:1" in options
    merged = slurm_options(args, main, "/owned", "ALL", mode="campaign-merge", after_job="123")
    assert not any(value.startswith(("--array=", "--gres=")) for value in merged)
    args.array_concurrency = 99
    assert "--array=0-8%9" in slurm_options(args, main, "/owned", "ALL")
    args.array_concurrency = 0
    with pytest.raises(ValueError, match="positive"):
        slurm_options(args, main, "/owned", "ALL")


def test_array_adjustment_checks_owner_and_quota_without_resubmission(tmp_path, monkeypatch):
    pytest.importorskip("paramiko")
    from scripts import remote_gpu

    receipt = {"mode": "campaign-array", "array_tasks": [0, 1, 2], "jobs": ["123"],
               "remote_run": "/data1/dsl01/campaigns/.mecar-flybrain-test",
               "archive_sha256": "sealed"}
    (tmp_path / "submission.json").write_text(json.dumps(receipt))
    before = ("JobId=123 ArrayJobId=123 ArrayTaskThrottle=1 UserId=dsl01(1) "
              "Account=ug QOS=normal JobName=mecar-flybrain-campaign-array "
              "WorkDir=/mnt/data1/dsl01/campaigns/.mecar-flybrain-test")
    state = {"description": before}
    commands = []

    def fake_execute(_ssh, command):
        commands.append(command)
        if command == "scontrol show job 123":
            return state["description"]
        if command.startswith("sacctmgr"):
            return "normal|cpu=32,gres/gpu=2,mem=128G|"
        assert command == "scontrol update JobId=123 ArrayTaskThrottle=2"
        state["description"] = before.replace("Throttle=1", "Throttle=2")
        return ""

    monkeypatch.setattr(remote_gpu, "execute", fake_execute)
    monkeypatch.setattr(remote_gpu, "connect", lambda: nullcontext(None))
    monkeypatch.setattr(remote_gpu, "read_setting", lambda *_args: "dsl01")
    args = argparse.Namespace(output=tmp_path, array_concurrency=2)
    remote_gpu.adjust_array(args)
    assert json.loads((tmp_path / "submission.json").read_text()) == receipt
    assert json.loads((tmp_path / "scheduler-adjustments.json").read_text())[0][
        "sealed_archive_sha256"] == "sealed"
    state["description"] = before.replace("WorkDir=/mnt", "WorkDir=/foreign")
    with pytest.raises(ValueError, match="ownership"):
        remote_gpu.adjust_array(args)
    state["description"] = before
    args.array_concurrency = 3
    with pytest.raises(ValueError, match="QoS"):
        remote_gpu.adjust_array(args)
    assert sum(command.startswith("scontrol update") for command in commands) == 1


def test_collector_promotes_with_main_policy_and_preserves_launch_receipt(tmp_path, monkeypatch):
    from scripts import watch_local_screening

    source, root, output = tmp_path / "source", tmp_path / "root", tmp_path / "output"
    pipeline = source / "runs/rdna4-pipeline"
    pipeline.mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    actor = root / "src/actor.py"
    actor.write_text("sensor-only actor fixture")
    (pipeline / "manifest.json").write_text(json.dumps({
        "source_sha256": {"src/actor.py": sha256(actor)},
    }))
    (pipeline / "remote-addition-config.json").write_text(json.dumps(proposal()))
    (pipeline / "result.json").write_text(json.dumps({
        "status": "completed", "candidates": ["a", "b", "c"],
    }))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(main_policy()))

    def fake_git(arguments, **kwargs):
        return "" if "--porcelain" in arguments else "sealed-revision\n"

    calls = []
    monkeypatch.setattr(watch_local_screening.subprocess, "check_output", fake_git)
    monkeypatch.setattr(watch_local_screening.subprocess, "run",
                        lambda command, **kwargs: calls.append(command))
    from scripts import verify_remote_preflight

    proofs = iter((None, {"status": "passed", "job": "123"}))
    monkeypatch.setattr(verify_remote_preflight, "verify_preflight", lambda *_: next(proofs))
    sleeps = []
    monkeypatch.setattr(watch_local_screening.time, "sleep", sleeps.append)
    watch_local_screening.watch(source, output, root, tmp_path / "wheels", True, policy_path,
                                preflight_output=tmp_path / "probe")
    assert sleeps == [30]
    assert json.loads((output / "remote-preflight-verified.json").read_text())["job"] == "123"
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--array-concurrency") + 1] == "2"
    assert command[command.index("--time") + 1] == "03:00:00"
    main = json.loads((output / "main-training-config.json").read_text())
    assert main["updates"] == 128 and main["main_training"]["task_count"] == 9
    plan = json.loads((output / "remote-addition-plan.json").read_text())
    assert plan["promotion_policy_sha256"] == sha256(policy_path)
    assert plan["main_selected"] == ["a", "c"]
    with pytest.raises(ValueError, match="before remote promotion"):
        watch_local_screening.watch(source, output, root, tmp_path / "wheels", True,
                                    policy_path, resume=True)


def test_collector_resume_reads_korean_policy_path_as_utf8(tmp_path):
    from scripts.watch_local_screening import watch

    source, output = tmp_path / "source", tmp_path / "output"
    pipeline = source / "runs/rdna4-pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "result.json").write_text('{"status":"failed","candidates":[]}',
                                          encoding="utf-8")
    output.mkdir()
    (output / "heartbeat.json").write_text(json.dumps({
        "collected_runs": ["full_mixed/seed-4101"], "promotion_policy": "학교/정책.json",
    }, ensure_ascii=False), encoding="utf-8")
    watch(source, output, tmp_path / "root", tmp_path / "wheels", False, resume=True)
    assert json.loads((output / "heartbeat.json").read_text(encoding="utf-8"))["state"] == "failed"


def test_collected_evidence_can_resume_without_a_running_wsl_instance(tmp_path, monkeypatch):
    from scripts import watch_local_screening

    source, output, root = tmp_path / "source", tmp_path / "output", tmp_path / "root"
    pipeline = source / "runs/rdna4-pipeline"
    pipeline.mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    actor = root / "src/actor.py"
    actor.write_text("unchanged actor")
    (pipeline / "manifest.json").write_text(json.dumps({
        "source_sha256": {"src/actor.py": sha256(actor)},
    }))
    (pipeline / "remote-addition-config.json").write_text(json.dumps(proposal()))
    (pipeline / "result.json").write_text('{"status":"completed","candidates":["a"]}')
    watch_local_screening.watch(source, output, root, tmp_path / "wheels", False)
    receipt_hash = sha256(output / "collection-verified.json")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(main_policy()))
    monkeypatch.setattr(watch_local_screening.subprocess, "check_output",
                        lambda args, **kw: "" if "--porcelain" in args else "revision\n")
    calls = []
    monkeypatch.setattr(watch_local_screening.subprocess, "run",
                        lambda args, **kw: calls.append(args))
    watch_local_screening.watch(tmp_path / "offline-wsl", output, root, tmp_path / "wheels",
                                True, policy, resume=True, collected_only=True)
    assert len(calls) == 1
    assert sha256(output / "collection-verified.json") == receipt_hash
    rechecked = json.loads((output / "collection-rechecked.json").read_text())
    assert rechecked["verified_files"] == 3
    assert rechecked["receipt_sha256"] == receipt_hash


def test_changed_collected_inputs_cannot_be_promoted(tmp_path):
    from scripts.watch_local_screening import verify_collected

    pipeline = tmp_path / "rdna4-pipeline"
    pipeline.mkdir()
    receipt = {"rdna4-pipeline": {}}
    for name in ("manifest.json", "result.json", "remote-addition-config.json"):
        (pipeline / name).write_text("original input")
        receipt["rdna4-pipeline"][name] = sha256(pipeline / name)
    (tmp_path / "collection-verified.json").write_text(json.dumps(receipt))
    (pipeline / "remote-addition-config.json").write_text("changed input")
    with pytest.raises(RuntimeError, match="evidence changed"):
        verify_collected(tmp_path)
