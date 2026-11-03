import numpy as np

from flybrain_drive.flight_long_stationary_reference import LongStationaryMotionReference
from test_causal_geometry_visual_port import sample


def test_requires_one_second_and_uses_only_last_three_stationary_features():
    reference = LongStationaryMotionReference()
    for i in range(25):
        features = np.full(6, i, dtype=np.float32)
        assert not reference.observe(features, sample(i), capture_id=i, decision_time=10.+.04*i+.001)
        assert reference.value is None
    assert reference.observe(np.full(6, 25, dtype=np.float32), sample(25),
                             capture_id=25, decision_time=11.001)
    np.testing.assert_allclose(reference.value, np.full(6, 24.))


def test_motion_and_time_gap_restart_the_uninterrupted_window():
    reference = LongStationaryMotionReference()
    features = np.zeros(6, dtype=np.float32)
    for i in range(15):
        assert not reference.observe(features, sample(i), capture_id=i, decision_time=10.+.04*i+.001)
    assert not reference.observe(features, sample(15, speed=1.), capture_id=15, decision_time=10.601)
    assert not reference.rows
    for i in range(16, 30):
        assert not reference.observe(features, sample(i), capture_id=i, decision_time=10.+.04*i+.001)
    assert reference.value is None
    assert not reference.observe(features, sample(35), capture_id=35, decision_time=11.401)
    assert len(reference.rows) == 1
    reference.reset()
    assert not reference.rows and reference.value is None
