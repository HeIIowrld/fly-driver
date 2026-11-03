"""Synthetic recurrent-cell tests for training-only policy preservation, not driving evidence."""

from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.multiview_learning import episode_chunk, replay_chunk
from flybrain_drive.policy_anchor import (
    anchored_selection_score,
    policy_anchor_loss,
    prepare_policy_anchors,
    replay_anchor_mse,
)


class Cell(torch.nn.Module):
    circuit = SimpleNamespace(size=1)

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(.2))

    def initial_state(self, batch):
        return self.gain.new_zeros(batch, 1)

    def prepare_drive(self, observation):
        return observation['x']*self.gain

    def advance(self, drive, state):
        state = .5*state+drive
        return state.repeat(1, 3), state


def episode(n=5, key='a'*64):
    return dict(source={'sha256': key}, observation={'x': torch.ones(n, 1)*.2},
                action=torch.zeros(n, 3), action_valid=torch.ones(n, 3, dtype=torch.bool))


def test_frozen_targets_preserve_full_episode_state_and_ignore_teacher_actions():
    model, data = Cell(), episode()
    targets = prepare_policy_anchors(model, [data], ['a'*64], 2, 1)
    obs, _, _, active = episode_chunk([data], 0, 5)
    expected, _ = replay_chunk(model, obs, active, model.initial_state(1), 1)
    torch.testing.assert_close(targets['a'*64], expected[:, 0])
    assert not targets['a'*64].requires_grad and model.gain.requires_grad
    data['action'].fill_(.9)
    altered = prepare_policy_anchors(model, [data], ['a'*64], 3, 1)
    torch.testing.assert_close(targets['a'*64], altered['a'*64])
    assert replay_anchor_mse(model, [data], targets, 2, 1) == 0.


def test_penalty_normalization_and_gradients_exclude_padding_and_unselected_episodes():
    data, other = episode(), episode(2, 'b'*64)
    anchors = prepare_policy_anchors(Cell(), [data, other], ['a'*64], 2, 1)
    prediction = torch.zeros(4, 2, 3, requires_grad=True)
    policy_anchor_loss(prediction, [data, other], 4, anchors).backward()
    assert prediction.grad[0, 0].abs().sum() > 0
    assert prediction.grad[1:, 0].count_nonzero() == 0
    assert prediction.grad[:, 1].count_nonzero() == 0
    shifted = anchors['a'*64][:, None]+.1
    total = sum(policy_anchor_loss(shifted[i:i+2], [data], i, anchors) for i in range(0, 5, 2))
    assert float(total) == pytest.approx(.01)


def test_selection_cannot_win_by_forgetting_declared_successful_policy():
    baseline = dict(rmse=[.1]*3, supervision_counts=[5]*3, policy_anchor_mse=0.)
    forgetting = dict(rmse=[.09]*3, supervision_counts=[5]*3, policy_anchor_mse=.1)
    assert anchored_selection_score(forgetting, 1.) > anchored_selection_score(baseline, 1.)
    with pytest.raises(ValueError, match='preservation score'):
        anchored_selection_score(dict(rmse=[.1]*3, supervision_counts=[5]*3), 1.)
    with pytest.raises(ValueError, match='exactly once'):
        prepare_policy_anchors(Cell(), [episode()], ['b'*64], 2, 1)


def test_anchor_metric_reuses_existing_batched_replay_without_extra_brain_ticks():
    from flybrain_drive.multiview_learning import replay_metrics

    class CountingCell(Cell):
        advances = 0

        def advance(self, drive, state):
            self.advances += 1
            return super().advance(drive, state)

    model, data, other = CountingCell(), episode(), episode(2, 'b'*64)
    anchors = prepare_policy_anchors(model, [data, other], ['a'*64], 2, 1)
    with torch.no_grad():
        model.gain.fill_(.3)
    expected = replay_anchor_mse(model, [data, other], anchors, 2, 1)
    model.advances = 0
    metrics = replay_metrics(model, [data, other], 2, 1,
        chunk_metric=lambda prediction, batch, offset:
            policy_anchor_loss(prediction, batch, offset, anchors))
    assert model.advances == 6
    assert metrics['extra_chunk_metric'] == pytest.approx(expected)
