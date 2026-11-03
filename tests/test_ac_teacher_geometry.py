"""Privileged geometry conventions, not a fly actor's perception algorithm."""

import numpy as np
import pytest

from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry, legacy_signed_gap
from flybrain_drive.graph import sha256


def test_legacy_gap_keeps_wrap_pivot_and_signed_segment_priority():
    line = np.array([[0, 0], [10, 0], [20, 0]])
    assert legacy_signed_gap([10, 1], line) == pytest.approx(-1)
    assert legacy_signed_gap([10, -1], line) == pytest.approx(1)
    # Legacy nearest pivot0 uses last->first as its earlier segment, even on an open line.
    assert legacy_signed_gap([5, 1], line) == pytest.approx(1)
    with pytest.raises(ValueError, match='Finite'):
        legacy_signed_gap([float('nan'), 1], line)


def test_map_rays_use_declared_pose_and_hashes_without_actor_images(tmp_path):
    track, reference = tmp_path / 'track.csv', tmp_path / 'reference.csv'
    x = np.linspace(-100, 100, 31)
    np.savetxt(track, np.c_[x, x * 0 + 5, x, x * 0 - 5], delimiter=',',
               header='left_border_x,left_border_y,right_border_x,right_border_y', comments='')
    np.savetxt(reference, [[-100, 0], [0, 0], [100, 0]], delimiter=',',
               header='pos_x,pos_y', comments='')
    geometry = PrivilegedTrackGeometry(track, reference, track_sha256=sha256(track),
                                       reference_sha256=sha256(reference))
    features = geometry.features(0, 0, 0)
    assert features['gap'] == 0
    assert len(features['sensors']) == 11
    np.testing.assert_allclose(features['sensors'][[0, 5, 10]], [5, 200, 5])
    with pytest.raises(ValueError, match='hashes'):
        PrivilegedTrackGeometry(track, reference, track_sha256='0' * 64,
                                reference_sha256=sha256(reference))
