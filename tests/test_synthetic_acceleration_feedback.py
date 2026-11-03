import pytest
import torch

from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.synthetic_acceleration_feedback import SyntheticAccelerationBodyFeedbackAdapter


def test_existing_projection_is_exactly_preserved_at_zero_acceleration_weights():
    torch.manual_seed(7)
    old = CenteredSpeedBodyFeedbackAdapter(5)
    new = SyntheticAccelerationBodyFeedbackAdapter(5)
    with torch.no_grad():
        old.projection.weight.normal_()
        new.projection.weight.copy_(old.projection.weight)
    body = torch.randn(12, 21)
    mask = torch.rand(12, 21) > .3
    assert torch.equal(new(body, mask), old(body, mask))


def test_acceleration_feedback_respects_per_axis_mask_and_bounds():
    model = SyntheticAccelerationBodyFeedbackAdapter(3)
    with torch.no_grad():
        model.acceleration_projection.weight.copy_(torch.eye(3))
    body = torch.zeros(1, 21)
    body[0, 5:8] = torch.tensor([3., -4., 5.])
    mask = torch.ones(1, 21, dtype=torch.bool)
    all_valid = model(body, mask)
    mask[0, 6] = False
    masked = model(body, mask)
    assert all_valid[0, 1] != 0 and masked[0, 1] == 0
    assert torch.equal(masked[:, [0, 2]], all_valid[:, [0, 2]])
    assert torch.all(torch.abs(masked) < .5)
    mask[:, 5:8] = False
    assert torch.equal(model(body, mask), torch.zeros(1, 3))


def test_nonfinite_and_wrong_mask_are_rejected():
    model = SyntheticAccelerationBodyFeedbackAdapter(3)
    body = torch.zeros(1, 21)
    with pytest.raises(ValueError): model(body, torch.ones(1, 21))
    body[0, 5] = float('nan')
    with pytest.raises(ValueError): model(body, torch.ones(1, 21, dtype=torch.bool))
