import numpy as np
import torch

from scripts.train_column_preview_comparison import ColumnBoundary, metrics


def test_masked_chunk_accumulation_matches_whole_history_gradient():
    torch.manual_seed(5)
    initial = torch.rand(16, 721)/721
    model = ColumnBoundary(initial, torch.zeros(896), torch.ones(896), True)
    x, y = torch.rand(5, 4, 14, 721), torch.randn(5, 6)
    valid = torch.tensor([[True]*6, [False, True]*3, [True]*6,
                          [True, False]*3, [True]*6])
    weights = valid/(6*valid.sum(0))
    ((model(x)-y).square()*weights).sum().backward()
    original = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for i in range(0, 5, 2):
        ((model(x[i:i+2])-y[i:i+2]).square()*weights[i:i+2]).sum().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, original[name], rtol=1e-4, atol=1e-7)
    assert model.spatial.grad.abs().sum() > 0
    fixed = ColumnBoundary(initial, torch.zeros(896), torch.ones(896), False)
    fixed(x).sum().backward()
    assert fixed.spatial.grad is None
    torch.testing.assert_close(fixed.spatial, initial)


def test_metrics_preserve_invalid_targets_and_original_clock():
    y = np.zeros((4, 6))
    y[1] = 100
    valid = np.ones((4, 6), bool)
    valid[1] = False
    result = metrics(y, valid, np.array([10., 11., 13., 14.]), np.zeros_like(y), np.zeros(6))
    assert result['full'][0]['count'] == 3
    assert result['after3s'][0]['count'] == 2
    assert result['full'][0]['rmse'] == 0
