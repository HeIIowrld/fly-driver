"""Sealed teacher data and numerical recurrent carry contracts."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import (
    condition_native_steering,
    configure_trainable_groups,
    episode_batches,
    episode_chunk,
    load_episodes,
    mixed_legacy_control,
    replay_chunk,
    replay_metrics,
    replay_output_activity,
    supervised_selection_score,
)
from flybrain_drive.multiview_sensory import MultiViewPacker


def test_native_selection_cannot_ignore_regressing_pedals_or_invent_missing_scores():
    baseline = dict(rmse=[.2, .2, .2], supervision_counts=[10, 10, 10])
    bad_pedals = dict(rmse=[.1, .5, .5], supervision_counts=[10, 10, 10])
    assert supervised_selection_score(bad_pedals) > supervised_selection_score(baseline)
    proxy = dict(rmse=[.2, None, None], supervision_counts=[10, 0, 0])
    assert supervised_selection_score(proxy) == pytest.approx(.04)
    with pytest.raises(ValueError, match='absent'):
        supervised_selection_score(dict(rmse=[.2, 0, None], supervision_counts=[10, 0, 0]))
    with pytest.raises(ValueError, match='No supervised'):
        supervised_selection_score(dict(rmse=[None] * 3, supervision_counts=[0] * 3))


def test_coast_loss_emphasizes_actual_zero_gas_labels_without_supervising_missing_heads():
    from flybrain_drive.multiview_learning import brake_label_weights, control_label_weights

    labels = torch.tensor([[0., 0., .2], [0., .2, 0.], [0., 0., 0.]])
    valid = torch.tensor([[True, True, True], [True, True, True], [True, False, True]])
    before = labels.clone(), valid.clone()
    torch.testing.assert_close(control_label_weights(labels, valid),
                               brake_label_weights(labels, valid))
    weights = control_label_weights(labels, valid, brake_weight=2, coast_weight=100)
    prediction = torch.full_like(labels, .1, requires_grad=True)
    ((prediction-labels).square()*weights).sum().backward()
    assert prediction.grad[0, 1] == pytest.approx(20.)
    assert prediction.grad[1, 1] == pytest.approx(-.2)
    assert prediction.grad[2, 1] == 0
    assert prediction.grad[0, 2] == pytest.approx(-.4)
    assert torch.equal(labels, before[0]) and torch.equal(valid, before[1])
    for weight in [0, float('nan'), 1001]:
        with pytest.raises(ValueError):
            control_label_weights(labels, valid, coast_weight=weight)


def test_coast_selection_uses_declared_weighted_objective_and_keeps_raw_rmse():
    metrics = dict(rmse=[.2, .2, .2], control_weighted_rmse=[.1, .1, .1],
                   supervision_counts=[10, 10, 10])
    assert supervised_selection_score(metrics) == pytest.approx(.01)
    assert metrics['rmse'] == [.2, .2, .2]


def test_native_steering_scale_uses_activity_without_changing_signs_and_rejects_dead_activity():
    readout = SimpleNamespace(output_size=4, steering_ports=torch.tensor([0, 1]),
        steering_signs=torch.tensor([-1., 1.]),
        steering_log_gain=torch.nn.Parameter(torch.tensor(50.)))
    activity = torch.tensor([[.1, .2, 0, 0], [.2, .1, 0, 0]])
    report = condition_native_steering(readout, activity, target_rms=.15, max_gain=1e8)
    assert report['achieved_rms'] == pytest.approx(.15)
    assert not report['teacher_labels_used'] and not report['output_ids_and_signs_changed']
    torch.testing.assert_close(readout.steering_signs, torch.tensor([-1., 1.]))
    previous = readout.steering_log_gain.detach().clone()
    with pytest.raises(ValueError, match='cannot reach'):
        condition_native_steering(readout, torch.zeros_like(activity), target_rms=.15, max_gain=1e8)
    torch.testing.assert_close(readout.steering_log_gain, previous)


def test_training_groups_never_unfreeze_recurrent_structure_and_optimizer_updates_selected_only():
    model = torch.nn.Module()
    model.adapter = torch.nn.Module()
    model.adapter.visual_adapter = torch.nn.Linear(2, 2, bias=False)
    model.adapter.organ_adapters = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    model.readout = torch.nn.Linear(2, 3, bias=False)
    model.circuit = torch.nn.Linear(2, 2, bias=False)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    selected = configure_trainable_groups(model, ['organs', 'readout'])
    optimizer = torch.optim.SGD(selected, lr=.1)
    x = model.adapter.visual_adapter(torch.ones(1, 2))
    x = model.circuit(x + model.adapter.organ_adapters[0](torch.ones(1, 2)))
    model.readout(x).square().sum().backward()
    optimizer.step()
    for name, p in model.named_parameters():
        frozen = name.startswith(('adapter.visual_adapter', 'circuit'))
        assert p.requires_grad != frozen
        if frozen:
            torch.testing.assert_close(p, original[name], rtol=0, atol=0)
        else:
            assert not torch.equal(p, original[name])
    with pytest.raises(ValueError, match='unique'):
        configure_trainable_groups(model, ['circuit'])
    configure_trainable_groups(model, ['visual'])
    assert [name for name, p in model.named_parameters() if p.requires_grad] == [
        'adapter.visual_adapter.weight']


def dataset(tmp_path, *, pedal=False, bad_mask=False):
    calibration = {"view_order": ["front", "rear"]}
    obs = {"view_rgb": np.ones((3, 2, 1, 3), np.float32) * .2,
           "view_contrast": np.zeros((3, 2, 1, 3), np.float32),
           "view_valid": np.ones((3, 2, 1), np.float32),
           "contrast_valid": np.zeros((3, 2, 1), np.float32),
           "view_age": np.zeros((3, 2), np.float32),
           "organs": np.zeros((3, 1), np.float32),
           "organ_valid": np.zeros((3, 1), np.float32)}
    if bad_mask:
        obs["view_valid"][0, 0, 0] = .5
    mask = np.zeros((3, 3), bool)
    mask[:, 0] = True
    mask[:, 1] = pedal
    path = tmp_path / "episode.npz"
    np.savez(path, **{"obs_" + k: v for k, v in obs.items()},
             action=np.zeros((3, 3)), action_valid=mask, observation_time=np.arange(3) * .04)
    manifest = {"schema": "flybrain.multiview.teacher.v1",
                "action_profile": "mujoco_curvature_proxy_only",
                "sensory_calibration": calibration, "actor_fields": sorted(MultiViewPacker.fields),
                "decisions": 3, "decision_dt_seconds": .04, "brain_ticks_per_decision": 4,
                "episodes": [{"episode_id": "a", "file": path.name,
                                               "sha256": sha256(path), "steps": 3}]}
    (tmp_path / "dataset.json").write_text(json.dumps(manifest))
    return calibration, manifest


def test_sealed_teacher_hash_path_and_proxy_pedal_guards(tmp_path):
    calibration, manifest = dataset(tmp_path)
    episodes, _ = load_episodes(tmp_path, calibration)
    assert len(episodes) == 1 and episodes[0]["action_valid"].sum() == 3
    manifest["episodes"][0]["file"] = "../outside.npz"
    (tmp_path / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path/hash"):
        load_episodes(tmp_path, calibration)
    dataset(tmp_path, pedal=True)
    with pytest.raises(ValueError, match="cannot supervise"):
        load_episodes(tmp_path, calibration)
    dataset(tmp_path, bad_mask=True)
    with pytest.raises(ValueError, match="binary"):
        load_episodes(tmp_path, calibration)
    dataset(tmp_path)
    with (tmp_path / "episode.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="path/hash"):
        load_episodes(tmp_path, calibration)


def test_optional_missing_source_requires_explicit_matching_calibration(tmp_path):
    calibration, manifest = dataset(tmp_path)
    path = tmp_path / 'episode.npz'
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in original.files}
    arrays['obs_view_valid'][:, 1] = 0
    arrays['obs_view_rgb'][:, 1] = 0
    np.savez(path, **arrays)
    manifest['episodes'][0]['sha256'] = sha256(path)
    (tmp_path / 'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='coverage'):
        load_episodes(tmp_path, calibration)
    calibration['required_views'] = ['front']
    (tmp_path / 'dataset.json').write_text(json.dumps(manifest))
    episodes, _ = load_episodes(tmp_path, calibration)
    assert not episodes[0]['observation']['view_valid'][:, 1].any()


class Integrator:
    circuit = SimpleNamespace(size=1)

    def initial_state(self, batch):
        return torch.zeros(batch, 1)

    def prepare_drive(self, observation):
        return observation["x"]

    def advance(self, drive, state):
        state = state + drive
        return state.repeat(1, 3), state


def test_output_activity_uses_last_neural_tick_excludes_padding_and_preserves_state():
    class WithReadout(Integrator):
        readout = torch.nn.Identity()

        def advance(self, drive, state):
            output, state = super().advance(drive, state)
            return self.readout(output), state

    episodes = [{'observation': {'x': torch.ones(n, 1)}, 'action': torch.zeros(n, 3),
                 'action_valid': torch.ones(n, 3, dtype=torch.bool)} for n in (5, 2)]
    model = WithReadout()
    activities = replay_output_activity(model, episodes, 3, 2)
    assert len(activities) == 7
    torch.testing.assert_close(activities[:, 0].sort().values,
                               torch.tensor([2., 2., 4., 4., 6., 8., 10.]))
    assert not model.readout._forward_pre_hooks


def test_chunked_replay_carries_state_and_preserves_ended_episode():
    episodes = [{"observation": {"x": torch.ones(n, 1)}, "action": torch.zeros(n, 3),
                 "action_valid": torch.ones(n, 3, dtype=torch.bool)} for n in (5, 2)]
    state = torch.zeros(2, 1)
    outputs = []
    for start in (0, 3):
        obs, _, valid, active = episode_chunk(episodes, start, 3)
        prediction, state = replay_chunk(Integrator(), obs, active, state.detach(), 2)
        outputs.append(prediction)
        assert valid.sum() == active.sum() * 3
    torch.testing.assert_close(state, torch.tensor([[10.], [4.]]))
    assert outputs[1][:, 1].count_nonzero() == 0
    assert outputs[1][2].count_nonzero() == 0


def test_dagger_student_executor_ignores_oracle_steering_and_unsupervised_pedals():
    teacher = np.array([-.25, .9], np.float32)
    student = np.array([-.4, 1., .8])
    command, chosen = mixed_legacy_control(teacher, student, 0, np.random.default_rng(3))
    np.testing.assert_allclose(command, [-.25, -.4])
    assert not chosen and teacher[1] == np.float32(.9)
    altered = teacher.copy()
    altered[1] = -.9
    command2, _ = mixed_legacy_control(altered, student, 0, np.random.default_rng(3))
    np.testing.assert_array_equal(command, command2)
    commanded, chosen = mixed_legacy_control(teacher, student, 1, np.random.default_rng(3))
    assert chosen
    np.testing.assert_array_equal(commanded, teacher)
    with pytest.raises(ValueError, match="Student execution"):
        mixed_legacy_control(teacher, None, .5, np.random.default_rng(3))


def test_merger_preserves_episode_bytes_and_rejects_contract_changes(tmp_path, monkeypatch):
    from scripts.merge_multiview_teacher import main

    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    calibration, a = dataset(left)
    _, b = dataset(right)
    merged = tmp_path / "merged"
    monkeypatch.setattr(sys, "argv", ["merge", "--input", str(left), "--input", str(right),
                                     "--output", str(merged)])
    main()
    episodes, manifest = load_episodes(merged, calibration)
    assert manifest["decisions"] == 6 and len(episodes) == 2
    assert len({e["source"]["episode_id"] for e in episodes}) == 2
    assert sha256(left / "episode.npz") == a["episodes"][0]["sha256"]
    assert sha256(right / "episode.npz") == b["episodes"][0]["sha256"]
    b["brain_ticks_per_decision"] = 3
    (right / "dataset.json").write_text(json.dumps(b))
    monkeypatch.setattr(sys, "argv", ["merge", "--input", str(left), "--input", str(right),
                                     "--output", str(tmp_path / "bad")])
    with pytest.raises(ValueError, match="equal sensory"):
        main()
    assert not (tmp_path / "bad").exists()


def test_length_batches_preserve_complete_episodes_and_replay_metrics():
    episodes = [{"observation": {"x": torch.ones(n, 1)}, "action": torch.zeros(n, 3),
                 "action_valid": torch.ones(n, 3, dtype=torch.bool)} for n in (5, 2, 5)]
    batches = episode_batches(episodes, 2)
    assert sorted(id(e) for batch in batches for e in batch) == sorted(map(id, episodes))
    assert all(len({len(e["action"]) for e in batch}) == 1 for batch in batches)
    together = replay_metrics(Integrator(), episodes, 3, 2)
    separated = replay_metrics(Integrator(), episodes, 3, 2, episodes_per_batch=1)
    assert separated == together
    with pytest.raises(ValueError, match="batch maximum"):
        episode_batches(episodes, 0)
