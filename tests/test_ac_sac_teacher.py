import hashlib
import warnings
from collections import OrderedDict

import numpy as np
import pytest
import torch

from flybrain_drive.ac_sac_teacher import (
    CHANNELS,
    PrivilegedSACTeacher,
    TeacherReferenceCurve,
    channel_scales,
    encode_teacher_observation,
    legacy_request_to_normalized_controls,
    relative_request,
)


def test_legacy_pedal_requests_normalize_independently_without_erasing_overlap():
    np.testing.assert_array_equal(legacy_request_to_normalized_controls([0, -1, -1]), [0, 0, 0])
    np.testing.assert_array_equal(legacy_request_to_normalized_controls([0, 0, 0]), [0, .5, .5])
    np.testing.assert_allclose(legacy_request_to_normalized_controls([-.2, .6, .8]), [-.2, .8, .9])
    with pytest.raises(ValueError, match='Bounded'):
        legacy_request_to_normalized_controls([0, 2, -1])


def fixture_state():
    state = dict(zip(CHANNELS, channel_scales("legacy_20241008"), strict=True))
    return {**state, "sensors": np.full(11, 200.), "out_of_track": 1,
            "steerAngle": 450., "accStatus": 1., "brakeStatus": 1.}


def test_teacher_feature_order_normalization_and_complete_history():
    row = fixture_state()
    obs = encode_teacher_observation(row, [row]*3, np.full(12, .1), profile="legacy_20241008")
    np.testing.assert_allclose(obs, np.ones(125))
    changed = dict(row, steerAngle=0., accStatus=.5, brakeStatus=.2)
    obs = encode_teacher_observation(changed, [row]*3, np.full(12, .1), profile="legacy_20241008")
    np.testing.assert_allclose(obs[47:50], [0., .5, .2])
    np.testing.assert_allclose(obs[38:47], np.ones(9))
    with pytest.raises(ValueError, match="history is unknown"):
        encode_teacher_observation(row, [], np.full(12, .1), profile="legacy_20241008")
    obs = encode_teacher_observation(row, [], np.full(12, .1),
                                     profile="legacy_20241008", allow_initial_padding=True)
    assert not obs[50:].any() and not obs[38:47].any()
    modern = encode_teacher_observation(row, [row]*3, np.full(12, .1), profile="current_2025")
    assert modern[9] == .5 and obs[9] == 1


def test_teacher_checkpoint_identity_distribution_heads_and_request_are_distinct(tmp_path):
    state = OrderedDict()
    for index, width, previous in [(0, 256, 125), (2, 256, 256), (4, 256, 256), (6, 6, 256)]:
        state[f"net.{index}.weight"] = torch.zeros((width, previous))
        state[f"net.{index}.bias"] = torch.zeros(width)
    state["net.6.bias"][:] = torch.tensor([-1., 0, 1, -30, 1, 30])
    checkpoint = tmp_path / "teacher.pt"
    torch.save(state, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    teacher = PrivilegedSACTeacher(checkpoint, digest)
    result = teacher.predict(np.ones((2, 125)))
    immutable = np.ones((2, 125))
    immutable.setflags(write=False)
    with warnings.catch_warnings(record=True) as caught:
        repeated = teacher.predict(immutable)
    assert not caught
    np.testing.assert_array_equal(repeated['deterministic_policy_action'],
                                  result['deterministic_policy_action'])
    np.testing.assert_allclose(result["deterministic_policy_action"],
                               np.tile(np.tanh([-1., 0, 1]), (2, 1)), rtol=1e-6)
    np.testing.assert_allclose(result["log_std"], np.tile([-20., 1., 2.], (2, 1)))
    request = relative_request([1, -1, 1], [.9, 0, 0], [.2, .48, .48], [-1]*3, [1]*3)
    np.testing.assert_allclose(request, [1, -.48, .48])
    with pytest.raises(ValueError, match="hash"):
        PrivilegedSACTeacher(checkpoint, "0"*64)
    with pytest.raises(ValueError, match="125"):
        teacher.predict(np.ones((2, 126)))
    assert all(not p.requires_grad for p in teacher.parameters())


def test_reference_segments_wrap_and_do_not_fill_missing_geometry(tmp_path):
    pytest.importorskip("scipy")
    t = np.linspace(0, 2*np.pi, 1000)
    path = tmp_path / "circle.csv"
    np.savetxt(path, np.c_[100*np.cos(t), 100*np.sin(t), np.full(len(t), .01)],
               delimiter=",", header="pos_x,pos_y,curvature", comments="")
    reference = TeacherReferenceCurve(path)
    np.testing.assert_allclose(reference.segment(reference.distance[-1]-10), np.full(12, .01))
    with pytest.raises(ValueError, match="domain"):
        reference.segment(-1)
