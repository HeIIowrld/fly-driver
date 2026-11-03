import pytest
import torch

from flybrain_drive.synthetic_acceleration_feedback import SyntheticAccelerationBodyFeedbackAdapter
from flybrain_drive.synthetic_load_feedback import SyntheticLoadBodyFeedbackAdapter


def sample():
    body = torch.zeros(2, 21)
    body[:, 8:12] = torch.tensor([[.25, .25, .25, .25], [.4, .1, .3, .2]])
    valid = torch.ones_like(body, dtype=torch.bool)
    return body, valid


def test_zero_initial_path_preserves_parent_with_existing_weights():
    torch.manual_seed(23)
    parent = SyntheticAccelerationBodyFeedbackAdapter(5)
    child = SyntheticLoadBodyFeedbackAdapter(5)
    with torch.no_grad():
        parent.projection.weight.normal_()
        parent.acceleration_projection.weight.normal_()
    child.load_state_dict(parent.state_dict(), strict=False)
    body, valid = sample()
    assert torch.equal(parent(body, valid), child(body, valid))


def test_uniform_load_neutral_and_native_order_distinguishable():
    child = SyntheticLoadBodyFeedbackAdapter(1)
    with torch.no_grad():
        child.load_projection.weight.copy_(torch.tensor([[1., -1., 0., 0.]]))
    body, valid = sample()
    out = child(body, valid)
    assert out[0, 0].item() == 0.
    assert out[1, 0].item() > 0.
    body[1, 8:10] = body[1, 8:10].flip(0)
    assert child(body, valid)[1, 0].item() < 0.


def test_partial_group_masks_added_load_path():
    child = SyntheticLoadBodyFeedbackAdapter(1)
    with torch.no_grad():
        child.load_projection.weight.fill_(2.)
        child.load_projection.weight[0, 0] = 4.
    body, valid = sample()
    valid[1, 8] = False
    assert child(body, valid)[1, 0].item() == 0.


def test_complete_malformed_load_rejected():
    child = SyntheticLoadBodyFeedbackAdapter(1)
    body, valid = sample()
    body[1, 8] = 1.2
    with pytest.raises(ValueError, match='load fractions'):
        child(body, valid)


def test_gradient_reaches_only_added_path_when_base_frozen():
    child = SyntheticLoadBodyFeedbackAdapter(2)
    child.projection.requires_grad_(False)
    child.acceleration_projection.requires_grad_(False)
    body, valid = sample()
    child(body, valid).sum().backward()
    assert child.load_projection.weight.grad.abs().sum().item() > 0
    assert child.projection.weight.grad is None
    assert child.acceleration_projection.weight.grad is None


@pytest.mark.parametrize("mask_bits", range(15))
def test_incomplete_load_never_leaks_via_existing_projection(mask_bits):
    torch.manual_seed(81)
    child = SyntheticLoadBodyFeedbackAdapter(3)
    with torch.no_grad():
        child.projection.weight.normal_()
        child.acceleration_projection.weight.normal_()
        child.load_projection.weight.normal_()
    body, valid = sample()
    for i in range(4):
        valid[:, 8 + i] = bool(mask_bits & (1 << i))
    changed = body.clone()
    changed[:, 8:12] = torch.tensor([.9, .05, .025, .025])
    assert torch.equal(child(body, valid), child(changed, valid))
