import numpy as np
import pytest

from flybrain_drive.compound_eye import (
    CompoundEyeProjector,
    EyeGeometry,
    PerspectiveView,
    synthetic_eye,
)
from flybrain_drive.sensor_contract import SampleTime


def geometry(rays, sigma=1e-6, side="left", origin=(0, 0, 0)):
    rays = np.asarray(rays, float)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return EyeGeometry(side, rays, np.full(len(rays), sigma), origin, "test synthetic geometry")


def project(image, eye, view=None, **kwargs):
    view = view or PerspectiveView(np.eye(3), [0, 0, 0], 90)
    return CompoundEyeProjector(eye, view).project(image, frame_id=0, time=SampleTime(0, 0),
                                                 **kwargs)


def test_calibrated_directions_preserve_left_up_and_image_aspect():
    # Independently analytic pinhole coordinates: u=(.5-y/x/(2*tan_h))*W-.5.
    image = np.zeros((101, 201, 3))
    image[:, :, 0] = np.linspace(0, 1, 201)
    image[:, :, 1] = np.linspace(0, 1, 101)[:, None]
    sample, coverage = project(image, geometry([[1, 0, 0], [1, .5, 0], [1, 0, .5]]))
    np.testing.assert_allclose(sample.response[:, 0], [.5, .37375, .5], atol=1e-8)
    np.testing.assert_allclose(sample.response[:, 1], [.5, .5, .2475], atol=1e-8)
    assert sample.valid.all()
    np.testing.assert_allclose(coverage, 1)


def test_acceptance_integrates_neighbors_and_does_not_clamp_blind_rays():
    image = np.zeros((101, 101, 3))
    image[50, 50] = 1
    narrow, _ = project(image, geometry([[1, 0, 0]]))
    broad, _ = project(image, geometry([[1, 0, 0]], sigma=3))
    assert narrow.response[0, 0] > .999
    assert broad.response[0, 0] == pytest.approx(4/9)
    blind, coverage = project(np.ones_like(image), geometry([[0, 1, 0], [-1, 0, 0]]))
    assert not blind.valid.any() and not blind.response.any() and not coverage.any()


def test_pixel_mask_and_view_origin_reject_unobserved_information():
    image = np.ones((21, 21, 3))
    valid = np.ones((21, 21), bool)
    valid[10, 10] = False
    sample, _ = project(image, geometry([[1, 0, 0]]), pixel_valid=valid)
    assert not sample.valid.any() and not sample.response.any()
    with pytest.raises(ValueError, match="depth"):
        CompoundEyeProjector(geometry([[1, 0, 0]], origin=(0, .1, 0)),
                             PerspectiveView(np.eye(3), [0, 0, 0], 90))


def test_pose_calibration_and_map_hash_are_explicit():
    yaw = np.deg2rad(30)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    eye = geometry([rotation[:, 0]])
    sample, _ = project(np.ones((21, 21, 3))*.4, eye,
                        PerspectiveView(rotation, [0, 0, 0], 90))
    np.testing.assert_allclose(sample.response, .4)
    assert eye.sha256 == sample.eye_map_sha256
    assert eye.sha256 != geometry([rotation[:, 0]], sigma=2).sha256
    assert "not_measured" in synthetic_eye("left").provenance
