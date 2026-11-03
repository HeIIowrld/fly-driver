import pytest
import torch

from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_sustained_opponent_boundary import SustainedOpponentBoundary


def test_zero_sustained_path_reproduces_original_and_invalid_views_fail():
    original = OpponentTemporalMotionBoundary(torch.zeros(576), torch.ones(576), 2)
    candidate = SustainedOpponentBoundary(torch.zeros(576), torch.ones(576), 2)
    with torch.no_grad():
        original.opponent_projection.weight.normal_(0, .02)
        candidate.opponent_projection.weight.copy_(original.opponent_projection.weight)
    views = torch.ones((1, 4), dtype=torch.bool)
    a, b = None, None
    for i in range(12):
        scene = torch.rand((1, 576))
        x, a = original.step(scene, views, a, None if i == 0 else .05)
        y, b = candidate.step(scene, views, b, None if i == 0 else .05)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    views[0, 2] = False
    with pytest.raises(ValueError, match="four valid"):
        candidate.step(scene, views, b, .05)


def test_constant_direction_survives_slow_path_and_preserves_opponent_sign():
    boundary = SustainedOpponentBoundary(torch.zeros(576), torch.ones(576), 1)
    with torch.no_grad():
        boundary.sustained_projection.weight[0, 0] = 1
        boundary.opponent_projection.weight[0, 0] = 1
    views = torch.ones((1, 4), dtype=torch.bool)
    zero = torch.zeros((1, 4, 9, 16))
    positive, negative, other = zero.clone(), zero.clone(), zero.clone()
    positive[0, 0, 2, 0] = 1
    negative[0, 0, 1, 0] = 1
    other[0, 1, 2, 0] = 1

    def replay(sample):
        _, state = boundary.step(zero.flatten(1), views, None, None)
        early, state = boundary.step(sample.flatten(1), views, state, .05)
        for _ in range(80):
            out, state = boundary.step(sample.flatten(1), views, state, .05)
        return early, out, state

    early, pos, state = replay(positive)
    _, neg, _ = replay(negative)
    _, wrong_view, _ = replay(other)
    assert early.item() == 0
    assert pos.item() > .38
    torch.testing.assert_close(pos, -neg)
    assert wrong_view.item() == 0
    assert (state[0] - state[1]).abs().max() < 1e-6
    pos.sum().backward()
    assert boundary.sustained_projection.weight.grad[0, 0] > 0
