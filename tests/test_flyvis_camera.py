import json

import numpy as np
import pytest

from flybrain_drive.flyvis_camera import VIEW_ORDER, FlyvisCameraProjector, integration_intervals


def projector():
    with open("configs/ac-csp-yaw-retina.json", encoding="utf8") as stream:
        config = json.load(stream)
    uv = np.array(
        [(q, r) for q in range(-15, 16) for r in range(max(-15, -15 - q), min(15, 15 - q) + 1)]
    )
    return FlyvisCameraProjector(config, uv[:, 0], uv[:, 1]), config, uv


def test_each_source_keeps_own_body_orientation_and_exact_reprojection():
    eye, config, uv = projector()
    center = np.flatnonzero((uv == 0).all(axis=1))[0]
    np.testing.assert_array_equal(
        eye.directions_body[:, center], [[1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]]
    )
    for index, name in enumerate(VIEW_ORDER):
        camera = config["views"][name]
        rays = eye.directions_body[index]
        forward = rays @ camera["forward_body"]
        screen_x = 0.5 * (1 + (rays @ camera["right_body"]) / forward * camera["projection_x"])
        screen_y = 0.5 * (1 - (rays @ camera["up_body"]) / forward * camera["projection_y"])
        np.testing.assert_allclose(np.column_stack([screen_x, screen_y]), eye.uv, atol=1e-14)


def test_horizontal_and_vertical_gradients_preserve_author_order_and_image_orientation():
    eye, _, uv = projector()
    horizontal = np.broadcast_to(np.arange(320) / 319 * 255, (180, 320)).astype(np.uint8)
    vertical = np.broadcast_to(np.arange(180)[:, None] / 179 * 255, (180, 320)).astype(np.uint8)
    for gradient, coordinate in [(horizontal, eye.uv[:, 0]), (vertical, eye.uv[:, 1])]:
        response = eye.sample(np.repeat(gradient[:, :, None], 3, axis=2))
        assert np.corrcoef(response, coordinate)[0, 1] > 0.9999
    with pytest.raises(ValueError, match="receptor order"):
        FlyvisCameraProjector(
            json.load(open("configs/ac-csp-yaw-retina.json", encoding="utf8")),
            uv[::-1, 0],
            uv[::-1, 1],
        )


def test_internal_ticks_preserve_fractional_camera_time_without_new_measurements():
    steps = integration_intervals(0.037123)
    assert max(steps) <= 0.005 and abs(sum(steps) - 0.037123) < 1e-14
    with pytest.raises(ValueError):
        integration_intervals(0.2)
