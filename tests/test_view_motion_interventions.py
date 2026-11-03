import pytest
import torch

from flybrain_drive.view_motion_interventions import mean_motion_views


def test_selected_motion_only_and_original_unchanged():
    scene = torch.arange(576, dtype=torch.float32).reshape(1, 576)
    original = scene.clone()
    mean = torch.full((576,), -3.0)
    result = mean_motion_views(scene, mean, (1, 3)).reshape(4, 9, 16)
    old = original.reshape(4, 9, 16)
    assert torch.equal(scene, original)
    assert torch.equal(result[:, 0], old[:, 0])
    assert torch.equal(result[0], old[0])
    assert torch.equal(result[2], old[2])
    assert torch.all(result[1, 1:] == -3)
    assert torch.all(result[3, 1:] == -3)


def test_invalid_view_or_shape_rejected():
    scene, mean = torch.zeros(1, 576), torch.zeros(576)
    for views in ((4,), (1, 1), (-1,)):
        with pytest.raises(ValueError):
            mean_motion_views(scene, mean, views)
    with pytest.raises(ValueError):
        mean_motion_views(scene[:, :-1], mean, (0,))
