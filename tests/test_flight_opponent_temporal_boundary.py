import pytest
import torch

from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary


def test_opponent_trace_keeps_view_polarity_and_uses_previous_input():
    boundary = OpponentTemporalMotionBoundary(torch.zeros(576), torch.ones(576), 1)
    with torch.no_grad():
        boundary.opponent_projection.weight.zero_()
        boundary.opponent_projection.weight[0, 0] = 1
    views = torch.ones(1, 4, dtype=torch.bool)
    zero = torch.zeros(1, 4, 9, 16)
    positive, negative, other_view = (zero.clone() for _ in range(3))
    positive[0, 0, 2, 0] = 1  # T4b
    negative[0, 0, 1, 0] = 1  # T4a
    other_view[0, 1, 2, 0] = 1

    def replay(sample):
        _, state = boundary.step(zero.flatten(1), views, None, None)
        immediate, state = boundary.step(sample.flatten(1), views, state, .05)
        later, _ = boundary.step(sample.flatten(1), views, state, .05)
        return immediate, later

    early, forward = replay(positive)
    _, backward = replay(negative)
    _, other = replay(other_view)
    assert early.item() == 0 and forward.item() > 0
    torch.testing.assert_close(forward, -backward)
    assert other.item() == 0
    forward.sum().backward()
    assert boundary.opponent_projection.weight.grad[0, 0] != 0
    assert boundary.projection.weight.grad is None


def test_new_boundary_starts_neutral_and_requires_all_views():
    boundary = OpponentTemporalMotionBoundary(torch.zeros(576), torch.ones(576), 2)
    scene = torch.rand(1, 576)
    views = torch.ones(1, 4, dtype=torch.bool)
    first, state = boundary.step(scene, views, None, None)
    second, _ = boundary.step(scene, views, state, .03)
    assert torch.count_nonzero(first) == torch.count_nonzero(second) == 0
    views[0, 3] = False
    with pytest.raises(ValueError, match='four valid'):
        boundary.step(scene, views, state, .03)
