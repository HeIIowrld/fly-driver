import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from flybrain_drive.parallel_learning import anchor_loss, train_parallel
from flybrain_drive.policy import policy_from_config
from flybrain_drive.training_validation import (
    TrainingValidator,
    ensure_test_separation,
    validation_score,
)
from scripts.run_campaign import run_campaign
from scripts.run_improvement_study import paired_gate, remote_proposal


def observation():
    return {
        "images": torch.ones(2, 2, 32, 32, 3, dtype=torch.uint8),
        "body": torch.ones(2, 38), "body_valid": torch.ones(2, 38),
        "camera_valid": torch.ones(2, 2), "camera_age": torch.zeros(2, 2),
    }


def test_value_head_training_cannot_change_actor_features_when_detached():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    shared = policy_from_config({})
    detached = policy_from_config({"critic_feature_grad": False})
    detached.load_state_dict(shared.state_dict())
    inputs = observation()
    a, av, _ = shared(inputs, shared.initial_state(2))
    b, bv, _ = detached(inputs, detached.initial_state(2))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(av, bv)
    (bv - 1).square().mean().backward()
    assert all(p.grad is None for p in detached.encoder.parameters())
    assert all(p.grad is None for p in detached.core.parameters())
    assert all(p.grad is None for p in detached.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in detached.critic.parameters())
    (av - 1).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in shared.encoder.parameters())


def test_anchor_preserves_speed_curvature_but_leaves_yaw_free():
    mean = torch.zeros(4, 2, 3, requires_grad=True)
    reference = torch.ones_like(mean)
    loss = anchor_loss(mean, reference, torch.tensor([1., 1., 0.]))
    loss.backward()
    assert mean.grad[..., :2].abs().sum() > 0
    assert mean.grad[..., 2].abs().sum() == 0


def test_validation_favors_completion_and_failure_reduction_over_distance():
    def result(completion, failure, progress):
        return {"aggregate": {"completion_rate": completion, "failure_rate": failure,
                              "mean_remaining_course_fraction": progress,
                              "mean_lateral_error_rms": 1}}
    assert validation_score(result(1, 0, .1)) > validation_score(result(0, 0, .9))
    assert validation_score(result(0, 0, .1)) > validation_score(result(0, 1, .9))
    with pytest.raises(ValueError, match="overlap"):
        ensure_test_separation({"suite": {"cases": [{"seed": 5}]}},
                               {"cases": [{"seed": 5}]})


def test_kl_rejection_restores_parameters_and_keeps_collector_history(tmp_path):
    config = {
        "device": "cpu", "num_envs": 2, "horizon": 4, "updates": 2,
        "epochs": 1, "torch_threads": 1, "checkpoint_interval": 1,
        "retain_checkpoint_interval": 1, "learning_rate": .1,
        "max_update_kl": 1e-10, "collector_restart": False,
        "environment": {"mode": "O", "max_steps": 100},
        "anchor_coefficient": 1,
    }
    run = train_parallel(config, tmp_path / "guard")
    report = json.loads((run / "training.json").read_text())
    assert report["parameter_change_l2"] == 0
    assert all(row["update_rejected"] for row in report["updates"])
    assert all(row["accepted_optimizer_steps"] == 0 for row in report["updates"])
    assert report["updates"][1]["episode_steps_at_start"] == [4, 4]
    assert all(value > 0 for value in report["updates"][1]["hidden_norm_at_start"])
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 2
    initial = torch.load(run / "initial-checkpoint.pt", weights_only=False)
    final = torch.load(run / "checkpoint.pt", weights_only=False)
    assert final["optimizer"]["state"] == initial["optimizer"]["state"]
    assert len(list((run / "checkpoints").glob('*.pt'))) == 2


def test_real_campaign_keeps_initial_last_and_development_selected_tests(tmp_path):
    config = {
        "device": "cpu", "num_envs": 2, "horizon": 3, "updates": 2,
        "epochs": 1, "torch_threads": 1, "critic_feature_grad": False,
        "critic_learning_rate": .001, "environment": {"mode": "O", "max_steps": 10},
        "validation": {"role": "development", "interval": 1, "record": False,
                       "suite": {"steps": 3, "cases": [{"name": "dev", "seed": 901}]}},
        "campaign": {"seeds": [71], "arms": [{"name": "head"}],
                     "evaluation": {"steps": 3, "cases": [{"name": "test", "seed": 902}]}},
    }
    output = tmp_path / "campaign"
    assert run_campaign(config, output) == 0
    row = json.loads((output / "summary.json").read_text())["runs"][0]
    assert all(key in row for key in ('initial', 'final', 'selected', 'checkpoint_selection'))
    history = row['checkpoint_selection']['history']
    assert [h['completed_updates'] for h in history] == [0, 1, 2]
    assert len({h['cases'][0]['seed'] for h in history}) == 1
    assert row['selected']['cases'][0]['seed'] == 902
    assert row['training']['completed_updates'] == 2


def test_regression_stop_requires_consecutive_development_failures(tmp_path, monkeypatch):
    def evaluation(progress):
        return {"aggregate": {"completion_rate": 0, "failure_rate": 1,
                               "mean_progress_delta": progress,
                               "mean_remaining_course_fraction": progress / 100,
                               "mean_lateral_error_rms": 1}, "cases": []}
    values = iter([evaluation(p) for p in (20, 5, 19, 4, 3)])
    monkeypatch.setattr('flybrain_drive.evaluation.evaluate_suite', lambda *a, **kw: next(values))
    config = {"validation": {"role": "development", "interval": 1,
                            "regression_patience": 2, "regression_ratio": .75,
                            "suite": {"cases": [{"name": "dev"}]}}}
    checkpoint = tmp_path / 'initial.pt'
    checkpoint.write_bytes(b'checkpoint')
    validator = TrainingValidator(config, tmp_path)
    assert [validator.evaluate(checkpoint, i) for i in range(5)] == [False] * 4 + [True]
    assert json.loads((tmp_path/'validation.json').read_text())['selected_completed_updates'] == 0


def test_promotion_requires_actual_last_policy_retention_not_selected_bc_fallback():
    def row(arm, progress, initial):
        return {"arm": arm, "seed": 71, "status": "completed",
                "initial": {"aggregate": {"mean_progress_delta": initial}},
                "final": {"aggregate": {"mean_progress_delta": progress, "failure_rate": 1},
                          "cases": [{"case": str(i), "progress_delta": progress}
                                    for i in range(4)]},
                "selected": {"aggregate": {"mean_progress_delta": 100}},
                "training": {"updates": [{"epochs_ran": 2, "accepted_optimizer_steps": 2}]}}
    control = row('control', 10, 30)
    candidate = row('candidate', 20, 30)
    assert not paired_gate(control, candidate)['passed']
    candidate['final']['aggregate']['mean_progress_delta'] = 29
    assert paired_gate(control, candidate)['passed']
    candidate['training']['updates'][0]['accepted_optimizer_steps'] = 0
    assert not paired_gate(control, candidate)['passed']
    candidate['seed'] = 72
    with pytest.raises(ValueError, match='same training seed'):
        paired_gate(control, candidate)


def test_remote_promotion_uses_fresh_seeds_data_and_separate_locked_tests():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/campaign-ppo-improvements.json').read_text(encoding='utf-8'))
    selected = [config['campaign']['arms'][2]]
    remote = remote_proposal(config, selected)
    assert remote['updates'] == 128 and remote['num_envs'] == 8
    assert remote['campaign']['seeds'] == [7201, 7202, 7203]
    assert [a['name'] for a in remote['campaign']['arms']] == ['ppo_control', 'head_critic_anchor']
    assert 'precollected_teacher' not in remote['behavior_cloning']
    ensure_test_separation(remote['validation'], remote['campaign']['evaluation'])
    assert remote['promotion']['no_local_checkpoint_transfer']
    assert 'precollected_teacher' in config['behavior_cloning']


def test_supervisor_reports_dead_launcher_instead_of_waiting_twelve_hours(tmp_path, monkeypatch):
    from scripts.watch_improvement_study import watch
    monkeypatch.setattr('scripts.watch_improvement_study.launcher_alive', lambda pid: False)
    output = tmp_path/'collection'
    with pytest.raises(RuntimeError, match='launcher exited'):
        watch(tmp_path/'absent-study', output, tmp_path, tmp_path, {}, launcher_pid=10)
    status = json.loads((output/'heartbeat.json').read_text())
    assert status['state'] == 'failed'


def test_development_selection_rejects_idle_policy_even_with_fewer_failures(tmp_path, monkeypatch):
    def evaluation(progress, failures):
        return {"aggregate": {"completion_rate": 0, "failure_rate": failures,
                               "mean_progress_delta": progress,
                               "mean_remaining_course_fraction": progress / 100,
                               "mean_lateral_error_rms": 0}, "cases": []}
    values = iter([evaluation(20, 1), evaluation(.1, 0)])
    monkeypatch.setattr('flybrain_drive.evaluation.evaluate_suite', lambda *a, **kw: next(values))
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'checkpoint')
    validator = TrainingValidator({"validation": {"role": "development", "suite": {
        "cases": [{"name": "dev"}]}}}, tmp_path)
    validator.evaluate(checkpoint, 0)
    validator.evaluate(checkpoint, 8)
    summary = json.loads((tmp_path/'validation.json').read_text())
    assert summary['selected_completed_updates'] == 0
    assert not summary['history'][-1]['progress_floor_passed']
