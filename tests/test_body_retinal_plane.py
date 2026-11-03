import numpy as np
import pytest
import torch

from flybrain_drive.body_retinal_plane import (
    fuse_scalar_views,
    level_rotation,
    plane_intersections,
    pool_on_plane,
)


def test_levelled_ray_hits_same_virtual_point_under_pitch_and_roll():
    origin = np.array([[2.5, 0., 1.]])
    point = np.array([8., 2., 0.])
    for pitch, roll in [(0., 0.), (.08, -.12), (-.05, .14)]:
        rotation = level_rotation([pitch], [roll])[0]
        direction = rotation.T@point-origin[0]
        direction /= np.linalg.norm(direction)
        xy, valid = plane_intersections(direction[None, None], origin, [pitch], [roll],
                                        np.array([True]))
        assert valid[0, 0, 0]
        np.testing.assert_allclose(xy[0, 0, 0], point[:2], atol=1e-12)
        np.testing.assert_allclose(rotation@rotation.T, np.eye(3), atol=1e-12)


def test_upward_parallel_outside_and_unknown_attitude_never_invent_support():
    rays = np.array([[[1., 0., 0.], [0., 0., 1.], [1., 0., -.001]]])
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    xy, valid = plane_intersections(rays, [[0, 0, 1]], [0.], [0.], np.array([True]))
    assert not valid.any() and np.isnan(xy).all()
    xy, valid = plane_intersections(np.array([[[0., 0., -1.]]]), [[0, 0, 1]], [0.], [0.],
                                    np.array([False]))
    assert not valid.any() and np.isnan(xy).all()


def test_scalar_fusion_uses_same_body_location_and_explicit_missing_mask():
    activity = torch.tensor([[[[2.]], [[4.]]]])
    xy = torch.tensor([[[[3., 1.]], [[3., 1.]]]])
    supported = torch.ones((1, 2, 1), dtype=torch.bool)
    centres = torch.tensor([[3., 1.], [20., 10.]])
    per_view, present = pool_on_plane(activity, xy, supported, centres, torch.ones(2))
    fused, valid = fuse_scalar_views(per_view, present)
    assert fused[0, 0, 0] == 3 and valid[0, 0]
    assert fused[0, 0, 1] == 0 and not valid[0, 1]
    supported[:, 1] = False
    per_view, present = pool_on_plane(activity, xy, supported, centres, torch.ones(2))
    fused, valid = fuse_scalar_views(per_view, present)
    assert fused[0, 0, 0] == 2 and valid[0, 0]


def test_invalid_retinal_geometry_rejected():
    with pytest.raises(ValueError):
        plane_intersections([[[2., 0., -1.]]], [[0, 0, 1]], [0.], [0.], np.array([True]))


def test_unknown_ray_geometry_remains_zero_with_false_support_after_fusion():
    values, present = pool_on_plane(torch.ones((1, 2, 1, 3)),
        torch.full((1, 2, 3, 2), float('nan')), torch.zeros((1, 2, 3), dtype=torch.bool),
        torch.zeros((2, 2)), torch.ones(2))
    fused, valid = fuse_scalar_views(values, present)
    assert torch.isfinite(fused).all() and not fused.any() and not valid.any()
