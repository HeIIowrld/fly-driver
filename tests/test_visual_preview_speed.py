import numpy as np
import pytest
import torch
from torch import nn

from flybrain_drive.flight_maneuver import ManeuverCurrents
from flybrain_drive.visual_preview_speed import PreviewSpeedCommand
from flybrain_drive.visual_yaw_bridge import RegisteredPreviewYaw


class Preview(nn.Module):
    def forward(self, visual, body):
        return torch.cat((visual[:, :5] * 0, visual[:, 384:385] * .02), dim=1)


def controller():
    preview = RegisteredPreviewYaw(Preview(), feature_mean=np.zeros(1024),
        feature_scale=np.ones(1024), body_mean=np.zeros(21), body_scale=np.ones(21),
        target_mean=np.zeros(6), target_scale=np.ones(6), body_route_indices=[0, 1, 4, 16])
    return PreviewSpeedCommand(preview, curvature_gain_m=100.)


def test_preview_changes_both_yaw_and_speed_and_reaches_motor_speed_error():
    model = controller()
    visual = torch.zeros(3, 1024)
    visual[:, 384] = torch.tensor([0., 1., -1.])
    body = torch.zeros(3, 21)
    body[:, 16] = 25.
    valid = torch.ones(3, 21, dtype=torch.bool)
    views = torch.ones(3, 4, dtype=torch.bool)
    goals = torch.tensor([[0., 1.]] * 3)
    motor = ManeuverCurrents([1, -1, 1, -1], 1)

    class FakeMotor:
        def drive(self, command, body, body_valid, racing_goals, **contracts):
            return motor(command, body, body_valid, racing_goals, **contracts)

    result = model.drive(FakeMotor(), visual, body, valid, views, goals)
    command = result['command']
    assert command[0, 1] > command[1, 1] and command[1, 1] == command[2, 1]
    assert command[1, 0] < 0 < command[2, 0]
    assert result['motor_drive'][1][0, 0] > result['motor_drive'][1][1, 0]
    assert (command[:, 1] <= 100 / 3.6).all()
    ablated = model(visual, body, valid, views, goals, intervention='motion_channels')
    torch.testing.assert_close(ablated['command'][:, 1], torch.full((3,), 100 / 3.6))
    views[1, 2] = False
    with pytest.raises(ValueError, match='Complete registered vision'):
        model(visual, body, valid, views, goals)


def test_invalid_gain_and_goal_fail_closed():
    for gain in (0, -1, float('nan'), True):
        with pytest.raises(ValueError, match='positive gain'):
            PreviewSpeedCommand(controller().preview, curvature_gain_m=gain)
    model = controller()
    with pytest.raises(ValueError, match='Complete registered vision'):
        model(torch.zeros(1, 1024), torch.zeros(1, 21), torch.ones(1, 21, dtype=torch.bool),
              torch.ones(1, 4, dtype=torch.bool), torch.tensor([[1., 1.]]))
