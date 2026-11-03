"""Experimental registered FlyVis preview to the existing physical motor boundary.

The readout predicts teacher-path *chord curvature*, not an optimal line. This
module is an engineering unit/sign conversion, not measured fly DN tuning or a
qualified driving policy. Only visual features, body sensing and a declared speed
task enter the actor; privileged route and teacher information are not inputs.
"""

import torch
from torch import nn

from .flight_maneuver import MANEUVER_CONTRACT
from .racing_goal import RACING_GOAL_CONTRACT


class RegisteredPreviewYaw(nn.Module):
    """Map the existing six-head registered preview to left-positive yaw rad/s.

    The predictor consumes the existing 1024 registered feature layout and 42
    masked body features. Its last head is screen-right chord curvature in 1/m.
    Measured unsigned body speed times curvature gives an approximate yaw-rate
    command when velocity follows the path without substantial sideslip. The sign
    is reversed for the physical circuit's left-positive convention. Feature
    moments and body/target calibration must come from the TRAIN-only fold.
    """

    def __init__(self, predictor, *, feature_mean, feature_scale, body_mean,
                 body_scale, target_mean, target_scale, body_route_indices):
        super().__init__()
        if not isinstance(predictor, nn.Module) or body_route_indices != [0, 1, 4, 16]:
            raise ValueError('Explicit registered predictor and original body route required')
        self.predictor = predictor
        for name, value, size in (
            ('feature_mean', feature_mean, 1024), ('feature_scale', feature_scale, 1024),
            ('body_mean', body_mean, 21), ('body_scale', body_scale, 21),
            ('target_mean', target_mean, 6), ('target_scale', target_scale, 6),
        ):
            array = torch.as_tensor(value, dtype=torch.float32).clone()
            if array.shape != (size,) or not torch.isfinite(array).all():
                raise ValueError('Finite original fold calibration required: ' + name)
            if name.endswith('_scale') and not torch.all(array > 0):
                raise ValueError('Positive original fold scales required: ' + name)
            self.register_buffer(name, array)
        route = torch.zeros(21, dtype=torch.bool)
        route[body_route_indices] = True
        self.register_buffer('body_route', route)

    def forward(self, registered_features, body, body_valid, view_valid,
                desired_speed_m_s, racing_goals, *, intervention=None):
        n = len(body)
        if (n < 1 or registered_features.shape != (n, 1024) or body.shape != (n, 21)
                or body_valid.shape != (n, 21) or body_valid.dtype != torch.bool
                or view_valid.shape != (n, 4) or view_valid.dtype != torch.bool
                or desired_speed_m_s.shape != (n,) or racing_goals.shape != (n, 2)
                or any(x.device != body.device for x in (registered_features, body_valid,
                       view_valid, desired_speed_m_s, racing_goals))
                or any(x.dtype != self.feature_mean.dtype for x in
                       (registered_features, body, desired_speed_m_s, racing_goals))
                or body.device != self.feature_mean.device
                or not all(torch.isfinite(x).all() for x in
                           (registered_features, body, desired_speed_m_s, racing_goals))
                or not view_valid.all() or not body_valid[:, [4, 16]].all()
                or torch.any(body[:, 16] < 0) or torch.any(desired_speed_m_s < 0)
                or torch.any(racing_goals[:, 0] != 0)
                or torch.any((racing_goals[:, 1] <= 0) | (racing_goals[:, 1] > 3.5))
                or torch.any(desired_speed_m_s > racing_goals[:, 1] * 100 / 3.6 + 1e-5)):
            raise ValueError('Complete registered vision, body validity and speed required')
        if intervention not in (None, 'motion_channels'):
            raise ValueError('Unknown registered visual intervention')
        visual = ((registered_features - self.feature_mean) / self.feature_scale).clamp(-5, 5)
        if intervention == 'motion_channels':
            visual = visual.clone()
            # Original registered_cpu: six fused scalar channels (384), four
            # views of eight motion channels (512), then support flags (128).
            visual[:, 384:896] = 0
        available = body_valid & self.body_route[None, :]
        standardized = ((body - self.body_mean) / self.body_scale).clamp(-5, 5)
        encoded_body = torch.cat((torch.where(available, standardized, 0),
                                  available.to(body.dtype)), dim=1)
        prediction = self.predictor(visual, encoded_body)
        if prediction.shape != (n, 6) or not torch.isfinite(prediction).all():
            raise ValueError('Finite original six-head preview required')
        curvature_right = prediction[:, 5] * self.target_scale[5] + self.target_mean[5]
        unclipped_yaw_left = -body[:, 16] * curvature_right
        yaw_left = unclipped_yaw_left.clamp(-MANEUVER_CONTRACT['maximum_absolute_yaw_rad_s'],
                                            MANEUVER_CONTRACT['maximum_absolute_yaw_rad_s'])
        command = torch.stack((yaw_left, desired_speed_m_s), dim=1)
        return dict(command=command, curvature_screen_right_per_m=curvature_right,
                    yaw_was_clipped=yaw_left != unclipped_yaw_left,
                    racing_goals=racing_goals)

    def drive(self, motor, registered_features, body, body_valid, view_valid,
              desired_speed_m_s, racing_goals, *, intervention=None):
        output = self(registered_features, body, body_valid, view_valid,
                      desired_speed_m_s, racing_goals, intervention=intervention)
        output['motor_drive'] = motor.drive(
            output['command'], body, body_valid, racing_goals,
            command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        return output
