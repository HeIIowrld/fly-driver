import json
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.flyvis_camera import VIEW_ORDER, FlyvisCameraProjector
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import (
    CalibratedVisualSelfMotion,
    StationaryMotionReference,
    VisualSelfMotionPool,
)


def geometry():
    with open("configs/ac-csp-yaw-retina.json", encoding="utf8") as stream:
        config = json.load(stream)
    uv = np.array(
        [(q, r) for q in range(-15, 16) for r in range(max(-15, -15 - q), min(15, 15 - q) + 1)]
    )
    projector = FlyvisCameraProjector(config, uv[:, 0], uv[:, 1])
    return config, projector.directions_body


def test_rotation_and_translation_templates_match_finite_perspective_motion():
    config, rays = geometry()
    pool = VisualSelfMotionPool(config, rays)
    epsilon = 1e-7
    for view, name in enumerate(VIEW_ORDER):
        camera = config["views"][name]
        ray = rays[view]

        def project(direction, camera=camera):
            denominator = direction @ camera["forward_body"]
            return 0.5 * np.column_stack(
                [
                    camera["projection_x"] * (direction @ camera["right_body"]) / denominator,
                    camera["projection_y"] * (direction @ camera["up_body"]) / denominator,
                ]
            )

        for mode in range(6):
            axis = np.eye(3)[mode % 3]
            if mode < 3:
                transformed = ray @ Rotation.from_rotvec(-axis * epsilon).as_matrix().T
            else:
                transformed = ray - epsilon * axis
                transformed /= np.linalg.norm(transformed, axis=-1, keepdims=True)
            numerical = (project(transformed) - project(ray)) / epsilon
            np.testing.assert_allclose(numerical, pool.templates[view, mode], rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_motion_pool_requires_finite_neural_channels_and_all_view_identities(bad_value):
    config, rays = geometry()
    pool = VisualSelfMotionPool(config, rays)
    with pytest.raises(ValueError):
        pool.features(np.zeros((3, 8, 721)))
    invalid = np.zeros((4, 8, 721))
    invalid[0, 0, 0] = bad_value
    with pytest.raises(ValueError):
        pool.features(invalid)


def native_sample(index, speed=0.0):
    raw = bytearray(308)
    struct.pack_into("<i", raw, 0, index)
    struct.pack_into("<f", raw, 28, speed)
    return decode_physics_prefix(bytes(raw), 10.0 + index * 0.04)


def test_reference_requires_stationary_body_unique_captures_and_explicit_reset():
    reference = StationaryMotionReference()
    for index in range(12):
        assert not reference.observe(
            np.ones(6),
            native_sample(index),
            capture_id=0,
            decision_time=10.0 + index * 0.04 + 0.001,
        )
    assert reference.value is None
    reference.reset()
    for index in range(11):
        reference.observe(
            np.ones(6),
            native_sample(index, speed=1.0),
            capture_id=index,
            decision_time=10.0 + index * 0.04 + 0.001,
        )
    assert reference.value is None
    for index in range(11):
        ready = reference.observe(
            np.ones(6),
            native_sample(index),
            capture_id=index,
            decision_time=10.0 + index * 0.04 + 0.001,
        )
    assert ready
    np.testing.assert_array_equal(reference.value, np.ones(6))
    reference.reset()
    assert reference.value is None


def test_decoder_rejects_failed_gate_and_requires_stationary_reference(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(dict(status="completed", calibration_gate_passed=False)), encoding="utf8"
    )
    with pytest.raises(ValueError, match="did not pass"):
        CalibratedVisualSelfMotion(tmp_path)
    np.savez(
        tmp_path / "calibration.npz",
        center=np.zeros(6),
        scale=np.ones(6),
        coefficients=np.vstack([np.eye(6), np.zeros((1, 6))]),
    )
    result.write_text(
        json.dumps(
            dict(
                status="completed",
                calibration_gate_passed=True,
                calibration_sha256=sha256(tmp_path / "calibration.npz"),
            )
        ),
        encoding="utf8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                config=dict(
                    plan=dict(reference_baseline_frames=[8, 11]),
                    camera_config_sha256=sha256("configs/ac-csp-yaw-retina.json"),
                )
            )
        ),
        encoding="utf8",
    )
    decoder = CalibratedVisualSelfMotion(tmp_path)
    with pytest.raises(ValueError, match="not ready"):
        decoder.decode(np.ones(6), reference=StationaryMotionReference())


def test_saved_motion_registration_rejects_reordered_views_rays_and_channels():
    config, rays = geometry()
    uv = np.array([(q, r) for q in range(-15, 16)
                   for r in range(max(-15, -15-q), min(15, 15-q)+1)])
    projector = FlyvisCameraProjector(config, uv[:, 0], uv[:, 1])
    registration = dict(u=uv[:, 0], v=uv[:, 1], image_uv=projector.uv.copy(),
                        body_directions=rays.copy(),
                        view_order=np.array(['front', 'left', 'right', 'rear']),
                        motion_types=np.array([f'{n}{d}' for n in ('T4', 'T5') for d in 'abcd']),
                        motion_indices=np.arange(8*721).reshape(8, 721))
    pool = VisualSelfMotionPool.from_registration(config, registration)
    np.testing.assert_array_equal(pool.templates, VisualSelfMotionPool(config, rays).templates)
    for name in ('view_order', 'motion_types', 'body_directions', 'motion_indices'):
        damaged = {key: value.copy() for key, value in registration.items()}
        if name == 'body_directions':
            damaged[name][0, [0, 1]] = damaged[name][0, [1, 0]]
        elif name == 'motion_indices':
            damaged[name][0, 0] = damaged[name][0, 1]
        else:
            damaged[name][[0, 1]] = damaged[name][[1, 0]]
        with pytest.raises(ValueError):
            VisualSelfMotionPool.from_registration(config, damaged)


def test_motion_node_identity_rejects_swapped_types_and_column_positions():
    uv = np.array([(q, r) for q in range(-15, 16)
                   for r in range(max(-15, -15-q), min(15, 15-q)+1)])
    names = [f'{n}{d}' for n in ('T4', 'T5') for d in 'abcd']
    reg = dict(u=uv[:, 0], v=uv[:, 1], motion_indices=np.arange(8*721).reshape(8, 721))
    types = np.repeat(names, 721)
    u, v = np.tile(uv[:, 0], 8), np.tile(uv[:, 1], 8)
    VisualSelfMotionPool.verify_motion_node_identities(reg, types, u, v)
    for a, b in ((0, 1), (0, 721)):
        wrong = {key: value.copy() for key, value in reg.items()}
        flat = wrong['motion_indices'].reshape(-1)
        flat[a], flat[b] = flat[b], flat[a]
        with pytest.raises(ValueError, match='type and lattice'):
            VisualSelfMotionPool.verify_motion_node_identities(wrong, types, u, v)
