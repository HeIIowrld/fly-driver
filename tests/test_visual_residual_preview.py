import pytest
import torch

from flybrain_drive.visual_residual_preview import ResidualPreview


def test_visual_residual_ignores_support_flags_and_zero_recovers_body():
    model = ResidualPreview().eval()
    with torch.no_grad():
        model.vision.weight[5, 384] = .5
    visual, body = torch.zeros(2, 1024), torch.randn(2, 42)
    body_only = model.body(body)
    torch.testing.assert_close(model(visual, body), body_only)
    visual[:, 900:] = 100
    torch.testing.assert_close(model(visual, body), body_only)
    visual[:, 384] = 1
    difference = model(visual, body) - body_only
    torch.testing.assert_close(difference[:, 5], torch.full((2,), .5))
    assert torch.count_nonzero(difference[:, :5]) == 0


def test_rejects_wrong_registered_feature_layout():
    with pytest.raises(ValueError, match='registered FlyVis'):
        ResidualPreview()(torch.zeros(2, 896), torch.zeros(2, 42))
