import pytest
import torch
from torch.nn import functional as F

from flybrain_drive.flight_commands import FrozenLinear
from flybrain_drive.flight_yaw_linear import zero_yaw_linear_forward


def test_zero_yaw_linear_preserves_original_weights_and_inputs_and_all_other_columns():
    torch.manual_seed(1230)
    layer = FrozenLinear(torch.randn(201, 6), torch.randn(201)).eval()
    weight, bias = layer.weight.clone(), layer.bias.clone()
    value = torch.randn(1200, 6)
    value[:, 3:] = torch.randint(0, 2, (1200, 3)).float()
    raw = value.clone()
    intervened = value.clone()
    intervened[:, 2] = 0.
    forward = zero_yaw_linear_forward(layer)
    expected = F.linear(intervened, layer.weight, layer.bias)
    torch.testing.assert_close(forward(value), expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(value, raw, atol=0, rtol=0)
    torch.testing.assert_close(layer.weight, weight, atol=0, rtol=0)
    torch.testing.assert_close(layer.bias, bias, atol=0, rtol=0)
    with torch.no_grad():
        layer.weight.add_(.01)
    with pytest.raises(ValueError, match='changed'):
        forward(value)


def test_optimization_refuses_trainable_or_different_body_encoder():
    with pytest.raises(ValueError, match='frozen'):
        zero_yaw_linear_forward(torch.nn.Linear(6, 201))
    with pytest.raises(ValueError, match='six-input'):
        zero_yaw_linear_forward(FrozenLinear(torch.randn(201, 7), torch.randn(201)).eval())
