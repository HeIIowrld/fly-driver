from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.flight_stop_vector_adapter import StopVectorAdapter
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.flight_visual_body_stop import (
    ExperimentalVisualBodyNarrowStop,
    ExperimentalVisualBodySpeedErrorStop,
    ExperimentalVisualBodyStop,
)


def test_stop_boundary_is_neutral_and_modulates_only_original_goal_ports(monkeypatch):
    base = torch.tensor([[.2, .3, .4, .5]])
    monkeypatch.setattr(ExperimentalVisualBodyAirflow, 'drive', lambda *args: base.clone())
    joint = ExperimentalVisualBodyStop.__new__(ExperimentalVisualBodyStop)
    torch.nn.Module.__init__(joint)
    joint.circuit = SimpleNamespace(backbone=SimpleNamespace(goal_ports=torch.tensor([1, 3])))
    joint.stop = StopVectorAdapter(2)

    def drive(speed):
        return joint.drive(None, None, None, None, torch.tensor([[0., speed]]),
                           None, None, None, None)

    torch.testing.assert_close(drive(0), base, rtol=0, atol=0)
    with torch.no_grad():
        joint.stop.stop_delta.fill_(1.)
    stopped, moving = drive(0), drive(1)
    assert torch.equal(stopped[:, [0, 2]], base[:, [0, 2]])
    assert torch.all(stopped[:, [1, 3]] > moving[:, [1, 3]])
    assert torch.all(moving[:, [1, 3]] > base[:, [1, 3]])


def test_narrow_gate_preserves_stop_but_suppresses_high_speed_delta():
    from flybrain_drive.flight_stop_narrow_adapter import NarrowStopVectorAdapter

    narrow, wide = NarrowStopVectorAdapter(2), StopVectorAdapter(2)
    with torch.no_grad():
        narrow.stop_delta.fill_(1.)
        wide.stop_delta.fill_(1.)
    speeds = torch.tensor([0., .5, 30. / 3.6])
    n, w = narrow(speeds), wide(speeds)
    torch.testing.assert_close(n[0], w[0], rtol=0, atol=0)
    assert torch.all(n[1:] < w[1:])
    assert torch.max(n[2]) < 1e-30
    assert issubclass(ExperimentalVisualBodyNarrowStop, ExperimentalVisualBodyStop)


def test_speed_feedback_changes_stop_gate_and_rejects_missing_speed(monkeypatch):
    base = torch.full((1, 4), .5)
    monkeypatch.setattr(ExperimentalVisualBodyAirflow, 'drive', lambda *args: base.clone())
    joint = ExperimentalVisualBodySpeedErrorStop.__new__(ExperimentalVisualBodySpeedErrorStop)
    torch.nn.Module.__init__(joint)
    joint.circuit = SimpleNamespace(backbone=SimpleNamespace(goal_ports=torch.tensor([1, 3])))
    joint.stop = StopVectorAdapter(2)
    with torch.no_grad():
        joint.stop.stop_delta.fill_(1.)
    body, valid = torch.zeros((1, 21)), torch.ones((1, 21), dtype=torch.bool)
    goals = torch.tensor([[0., .5 / (30. / 3.6)]])
    slow = joint.drive(None, None, body, valid, goals, None, None, None, None)
    body[:, 16] = 8.
    fast = joint.drive(None, None, body, valid, goals, None, None, None, None)
    assert torch.all(fast[:, [1, 3]] > slow[:, [1, 3]])
    assert torch.equal(fast[:, [0, 2]], base[:, [0, 2]])
    valid[:, 16] = False
    with pytest.raises(ValueError, match='vehicle speed'):
        joint.drive(None, None, body, valid, goals, None, None, None, None)
