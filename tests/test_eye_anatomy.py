import json

import numpy as np
import pytest

from flybrain_drive.eye_anatomy import load_anatomical_eyes, split_axes_by_lens
from flybrain_drive.graph import sha256


def data():
    return (np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0.]]), np.array([2, 3, 1]),
            np.array([1, 1, 0]), np.array([1, 0, 1]), np.eye(3))


def test_one_based_cone_permutation_and_side_assignment_preserve_actual_directions():
    arrays = split_axes_by_lens(*data())
    np.testing.assert_array_equal(arrays["left_directions_head"], [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(arrays["left_lens_rows"], [1, 2])
    np.testing.assert_array_equal(arrays["left_cone_rows"], [3, 1])
    values = list(data())
    values[1] = np.array([1, 1, 3])
    with pytest.raises(ValueError, match="permutation"):
        split_axes_by_lens(*values)
    values = list(data())
    values[3] = np.array([1, 1, 0])
    with pytest.raises(ValueError, match="sides disagree"):
        split_axes_by_lens(*values)


def test_virtual_mount_acceptance_and_file_hash_are_explicit(tmp_path):
    path = tmp_path / "axes.npz"
    np.savez_compressed(path, **split_axes_by_lens(*data()))
    path.with_suffix(".json").write_text(json.dumps({"geometry_sha256": sha256(path),
        "upstream_commit": "test", "source_kind": "toy_test_not_anatomical_measurement"}))
    origins = {"left": [0, .3, 0], "right": [0, -.3, 0]}
    eyes = load_anatomical_eyes(path, origins, sigma_degrees=1, head_to_body_rotation=np.eye(3))
    provenance = json.loads(eyes[0].provenance)
    assert provenance["acceptance_sigma_degrees_engineering_assumption"] == 1
    assert provenance["MaleCNS_retinal_neurons_registered"] == 0
    np.testing.assert_array_equal(eyes[0].directions_body, [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="proper"):
        load_anatomical_eyes(path, origins, sigma_degrees=1,
                             head_to_body_rotation=np.diag([-1, 1, 1]))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_anatomical_eyes(path, origins, sigma_degrees=1, head_to_body_rotation=np.eye(3))
