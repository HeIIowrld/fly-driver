import pytest
import torch

from flybrain_drive.retinal_residual_port import RetinalResidualPort


def test_zero_adapter_preserves_original_visual_current_exactly():
    adapter = RetinalResidualPort(torch.zeros(256), torch.ones(256), 42)
    base = torch.linspace(-8, 8, 42).reshape(1, -1)
    extra = torch.linspace(-1, 1, 256).reshape(1, -1)
    assert torch.equal((base+adapter(extra)).sigmoid(), base.sigmoid())


def test_new_visual_information_can_reach_existing_downstream_state():
    adapter = RetinalResidualPort(torch.zeros(256), torch.ones(256), 2)
    x = torch.ones(1, 256)
    # Explicitly synthetic neural coupling, not a measured circuit fixture.
    state = (torch.zeros(1, 2)+adapter(x)).sigmoid() @ torch.tensor([[1.], [-.5]])
    state.square().sum().backward()
    assert adapter.projection.weight.grad is not None
    assert float(adapter.projection.weight.grad.abs().sum()) > 0


def test_missing_wrong_dimension_or_nonfinite_retina_is_rejected():
    adapter = RetinalResidualPort(torch.zeros(256), torch.ones(256), 2)
    for x in [None, torch.zeros(1, 576), torch.full((1, 256), float('nan'))]:
        with pytest.raises(ValueError):
            adapter(x)


def test_modulation_is_bounded_and_calibration_not_aliased():
    ref, scale = torch.zeros(256), torch.ones(256)
    adapter = RetinalResidualPort(ref, scale, 2)
    ref.fill_(1)
    assert torch.equal(adapter.reference, torch.zeros(256))
    with torch.no_grad():
        adapter.projection.weight.fill_(10)
    assert torch.all(adapter(torch.ones(1, 256)).abs() <= .5)
