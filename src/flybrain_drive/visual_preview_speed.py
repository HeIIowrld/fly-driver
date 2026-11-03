"""Research-only visual preview speed command for the physical flight boundary.

The learned 20 m chord is not a road curvature measurement or optimal racing line.
The positive gain is an engineering hypothesis, not an identified fly circuit or
vehicle tire limit. This module is not a live controller or a braking guarantee.
"""

import math

import torch
from torch import nn

from .flight_maneuver import MANEUVER_CONTRACT
from .racing_goal import RACING_GOAL_CONTRACT
from .visual_yaw_bridge import RegisteredPreviewYaw


class PreviewSpeedCommand(nn.Module):
    """Use one visual prediction for both signed yaw and a bounded speed target."""

    def __init__(self, preview, *, curvature_gain_m):
        super().__init__()
        if (not isinstance(preview, RegisteredPreviewYaw)
                or isinstance(curvature_gain_m, bool)
                or not isinstance(curvature_gain_m, (float, int))
                or not math.isfinite(curvature_gain_m) or curvature_gain_m <= 0):
            raise ValueError('Registered preview and explicit positive gain in metres required')
        self.preview = preview
        self.curvature_gain_m = float(curvature_gain_m)

    def forward(self, registered_features, body, body_valid, view_valid, racing_goals, *,
                intervention=None):
        if (not isinstance(racing_goals, torch.Tensor) or racing_goals.ndim != 2
                or racing_goals.shape != (len(body), 2)):
            raise ValueError('One explicit racing ceiling per body sample required')
        ceiling_m_s = racing_goals[:, 1] * (100. / 3.6)
        output = self.preview(registered_features, body, body_valid, view_valid, ceiling_m_s,
                              racing_goals, intervention=intervention)
        curvature = output['curvature_screen_right_per_m']
        # A smooth dimensionless factor; no claim of measured friction or braking distance.
        fraction = torch.rsqrt(1. + self.curvature_gain_m * curvature.abs())
        desired = ceiling_m_s * fraction
        if not torch.isfinite(desired).all():
            raise ValueError('Finite preview-derived desired speed required')
        output['command'] = torch.stack((output['command'][:, 0], desired), dim=1)
        output['preview_speed_fraction'] = fraction
        return output

    def drive(self, motor, registered_features, body, body_valid, view_valid, racing_goals, *,
              intervention=None):
        output = self(registered_features, body, body_valid, view_valid, racing_goals,
                      intervention=intervention)
        output['motor_drive'] = motor.drive(
            output['command'], body, body_valid, racing_goals,
            command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        return output
