from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.multiview_learning import (
    brake_label_weights,
    replay_metrics,
    supervised_selection_score,
)


def test_rare_braking_gradient_is_emphasized_without_unmasking_or_changing_labels():
    target = torch.zeros(100, 3)
    target[0, 2] = .5
    target[1, 2] = .9
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1, 2] = False
    before = target.clone()
    prediction = torch.zeros_like(target, requires_grad=True)
    ordinary = brake_label_weights(target, valid)
    weighted = brake_label_weights(target, valid, 20.)
    assert torch.equal(ordinary, valid.float())
    assert weighted[1, 2] == 0
    assert torch.equal(weighted[:, :2], ordinary[:, :2])
    assert torch.equal(target, before)
    ordinary_gradient = torch.autograd.grad(
        ((prediction-target).square()*ordinary).sum(), prediction, retain_graph=True)[0]
    gradient = torch.autograd.grad(((prediction-target).square()*weighted).sum(), prediction)[0]
    assert gradient[0, 2] == ordinary_gradient[0, 2]*20
    assert gradient[1, 2] == 0


def test_weighted_training_selection_and_invalid_weight():
    failed = dict(rmse=[.1, .1, .03], brake_weighted_rmse=[.1, .1, .4],
                  supervision_counts=[100, 100, 100])
    learned = dict(rmse=[.11, .11, .04], brake_weighted_rmse=[.11, .11, .1],
                   supervision_counts=[100, 100, 100])
    assert supervised_selection_score(learned) < supervised_selection_score(failed)
    for weight in (0, float('nan'), 101):
        with pytest.raises(ValueError):
            brake_label_weights(torch.zeros(1, 3), torch.ones(1, 3), weight)


def test_replay_preserves_ordinary_metrics_and_excludes_padding_from_weighted_counts():
    class ZeroModel:
        circuit = SimpleNamespace(size=1)

        def initial_state(self, batch):
            return torch.zeros(batch, 1)

        def prepare_drive(self, observation):
            return observation['x']

        def advance(self, drive, state):
            return torch.zeros(len(state), 3), state

    episodes = [dict(observation={'x': torch.zeros(n, 1)}, action=torch.zeros(n, 3),
                     action_valid=torch.ones(n, 3, dtype=torch.bool)) for n in (4, 1)]
    episodes[0]['action'][0, 2] = .5
    episodes[0]['action'][1, 2] = .8
    episodes[0]['action_valid'][1, 2] = False
    ordinary = replay_metrics(ZeroModel(), episodes, 3, 1)
    weighted = replay_metrics(ZeroModel(), episodes, 3, 1, brake_positive_weight=20.)
    assert weighted['rmse'] == ordinary['rmse']
    assert weighted['supervision_counts'] == [5, 5, 4]
    assert weighted['weighted_supervision_counts'] == [5, 5, 23]
    assert weighted['brake_weighted_rmse'][2] == pytest.approx((5/23)**.5)
