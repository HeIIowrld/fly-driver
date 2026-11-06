import torch

from flybrain_drive.visual_descending_port import VisualDescendingPort


def make():
    return VisualDescendingPort(torch.zeros(320), torch.ones(320), torch.tensor([-1., 1., -1., 1.]))


def test_initial_tonic_is_balanced_and_shared_by_matched_control():
    a, b = make(), make()
    x = torch.ones(1, 320)
    assert torch.equal(a(x), b(x))
    assert torch.allclose(a(x), torch.full((1, 4), .001), atol=1e-9)


def test_zero_contrast_has_nonzero_learning_gradient():
    a = make()
    current = a(torch.ones(1, 320))
    (current[:, 1]-current[:, 0]).sum().backward()
    assert float(a.projection.weight.grad.abs().sum()) > 0


def test_contrast_reversal_exchanges_oriented_populations_and_bounds_currents():
    a = make()
    with torch.no_grad():
        a.projection.weight.fill_(.01)
    positive, negative = a(torch.ones(1, 320)), a(-torch.ones(1, 320))
    assert torch.allclose(positive[:, [0, 1]], negative[:, [1, 0]])
    assert bool((positive > 0).all() and (positive < .2).all())
