import pytest
import torch

from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
from test_unified_flight_sensorimotor import make_model, invoke


def test_centered_per_view_and_invalid_view_rejected():
    b = RetinalSpatialContrastBoundary(2)
    with torch.no_grad():
        b.projection.weight.copy_(torch.arange(128).reshape(2, 64) / 100.)
    torch.manual_seed(15)
    scene = torch.randn(1, 576)
    shifted = scene.reshape(1, 4, 9, 16).clone()
    shifted[:, :, 0, :] += torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1)
    valid = torch.ones(1, 4, dtype=torch.bool)
    torch.testing.assert_close(b(scene, valid), b(shifted.reshape(1, 576), valid), atol=3e-6, rtol=1e-6)
    valid[0, 2] = False
    with pytest.raises(ValueError):
        b(scene, valid)


def test_optional_zero_boundary_and_single_state_causal_effect():
    torch.manual_seed(9)
    baseline = make_model()
    boundary_model = make_model()
    boundary_model.load_state_dict(baseline.state_dict())
    boundary_model.spatial_contrast = RetinalSpatialContrastBoundary(2)
    scene = torch.zeros(1, 576)
    scene[0, 12] = 1.
    invoke(baseline, 1., scene)
    invoke(boundary_model, 1., scene)
    torch.testing.assert_close(baseline.previous_drive, boundary_model.previous_drive)
    baseline.reset_episode(); boundary_model.reset_episode()
    with torch.no_grad():
        boundary_model.spatial_contrast.projection.weight[0, 12] = 1.
    invoke(baseline, 1., scene); invoke(boundary_model, 1., scene)
    assert not torch.equal(baseline.previous_drive[:, :2], boundary_model.previous_drive[:, :2])
    invoke(baseline, 1.04, scene); invoke(boundary_model, 1.04, scene)
    assert not torch.equal(baseline.state[:, :2], boundary_model.state[:, :2])


def test_reference_is_centered_validated_and_zero_default_compatible():
    ref = torch.randn(4, 16)
    b = RetinalSpatialContrastBoundary(2, reference=ref)
    torch.testing.assert_close(b.reference.mean(-1), torch.zeros(4), atol=1e-6, rtol=0)
    assert 'reference' in b.state_dict()
    with pytest.raises(ValueError):
        RetinalSpatialContrastBoundary(2, reference=torch.full((4, 16), float('nan')))


def test_contrast_limit_is_causal_bounded_and_does_not_amplify_small_inputs():
    limited = RetinalSpatialContrastBoundary(2, max_contrast_rms=.1)
    plain = RetinalSpatialContrastBoundary(2)
    with torch.no_grad():
        limited.projection.weight.fill_(.1)
        limited.projection.weight[:, 0] = 1.
        plain.projection.weight.copy_(limited.projection.weight)
    scene = torch.zeros(1, 4, 9, 16)
    scene[:, 0, 0, 0] = .01
    valid = torch.ones(1, 4, dtype=torch.bool)
    torch.testing.assert_close(limited(scene.reshape(1,576),valid), plain(scene.reshape(1,576),valid))
    large = scene * 1000
    assert limited(large.reshape(1,576),valid).abs().max() < plain(large.reshape(1,576),valid).abs().max()
    assert 'max_contrast_rms' in limited.state_dict()
    with pytest.raises(ValueError):
        RetinalSpatialContrastBoundary(2, max_contrast_rms=0.)
