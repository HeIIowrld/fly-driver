import numpy as np
import pytest
import torch

from flybrain_drive.flight_geometry_fresh_hold import FreshEvolvingGeometryPort, FreshHoldGeometryPort
from test_causal_geometry_visual_port import make_port, sample


def prepared(cls=FreshHoldGeometryPort):
    old = make_port()
    port = cls(old.base, old.pool, 2)
    motion = np.ones((4, 8, 721), dtype=np.float32)
    valid = np.ones(4, dtype=bool)
    for i in range(11):
        now = 10. + .04*i + .001
        assert not port.observe(motion, valid, sample(i), capture_id=i,
                                decision_time=now, camera_receipt_time=now)
    motion[0, 1] = 2.
    assert port.observe(motion, valid, sample(11, speed=30), capture_id=11,
                        decision_time=10.441, camera_receipt_time=10.441)
    with torch.no_grad():
        port.projection.weight[0, 2] = 1.
    return port, motion, valid


def test_fresh_repeat_holds_but_is_consumed_and_expires():
    port, motion, valid = prepared()
    first = port(None, None, None)
    assert port.observe(motion, valid, sample(12), capture_id=11,
                        decision_time=10.481, camera_receipt_time=10.441)
    assert torch.equal(first, port(None, None, None))
    assert torch.equal(port(None, None, None), torch.tensor([[.2, -.1]]))
    assert not port.observe(motion, valid, sample(12), capture_id=11,
                            decision_time=10.492, camera_receipt_time=10.441)
    assert port.reason == 'camera_stale'
    assert port.current is None


def test_repeat_cannot_refresh_receipt_or_hide_invalid_view():
    port, motion, valid = prepared()
    with pytest.raises(ValueError, match='original receipt'):
        port.observe(motion, valid, sample(12), capture_id=11,
                     decision_time=10.481, camera_receipt_time=10.480)
    assert not port.observe(motion, np.array([True, False, True, True]), sample(12),
                            capture_id=11, decision_time=10.482, camera_receipt_time=10.441)
    assert port.current is None
    port.reset_episode()
    assert port.reference.value is None and port._held is None


def test_fresh_evolving_motion_changes_response_without_new_camera_capture():
    port, motion, valid = prepared(FreshEvolvingGeometryPort)
    old = port.current.clone()
    port(None, None, None)
    motion[0, 1] = 4.
    assert port.observe(motion, valid, sample(12), capture_id=11,
                        decision_time=10.481, camera_receipt_time=10.441)
    assert port.reason == 'fresh_capture_evolving_motion'
    assert not torch.equal(port.current, old)
    assert not port.observe(motion, valid, sample(12), capture_id=11,
                            decision_time=10.492, camera_receipt_time=10.441)
    assert port.current is None and port.reason == 'camera_stale'
