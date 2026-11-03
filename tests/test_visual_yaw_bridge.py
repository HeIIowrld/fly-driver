import numpy as np
import pytest
import torch
from torch import nn

from flybrain_drive.flight_maneuver import ManeuverCurrents
from flybrain_drive.visual_yaw_bridge import RegisteredPreviewYaw
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_visual_body_preview import body_features


def bridge(predictor, **changes):
    values = dict(feature_mean=np.zeros(1024), feature_scale=np.ones(1024),
                  body_mean=np.zeros(21), body_scale=np.ones(21),
                  target_mean=np.zeros(6), target_scale=np.ones(6),
                  body_route_indices=[0, 1, 4, 16])
    values.update(changes)
    return RegisteredPreviewYaw(predictor, **values)


def inputs(n=2):
    visual = torch.zeros(n, 1024)
    body = torch.zeros(n, 21)
    body[:, 16] = 10
    valid = torch.ones(n, 21, dtype=torch.bool)
    views = torch.ones(n, 4, dtype=torch.bool)
    return visual, body, valid, views, torch.full((n,), 10.), torch.tensor([[0., 1.]] * n)


class MotionPreview(nn.Module):
    def forward(self, visual, body):
        return torch.cat((visual[:, :5] * 0, visual[:, 384:385] * .01), dim=1)


def test_right_curvature_drives_left_positive_motor_with_motion_ablation():
    adapter = bridge(MotionPreview())
    visual, body, valid, views, speed, goals = inputs()
    visual[:, 384] = 1
    motor_boundary = ManeuverCurrents([1, -1, 1, -1], 1)

    class Motor:
        def drive(self, command, body, body_valid, racing_goals, **contracts):
            return motor_boundary(command, body, body_valid, racing_goals, **contracts)

    actual = adapter.drive(Motor(), visual, body, valid, views, speed, goals)
    removed = adapter.drive(Motor(), visual, body, valid, views, speed, goals,
                            intervention='motion_channels')
    torch.testing.assert_close(actual['command'][:, 0], torch.full((2,), -.1))
    assert torch.count_nonzero(actual['motor_drive'][0][:, [1, 3]]) == 4
    assert torch.count_nonzero(actual['motor_drive'][0][:, [0, 2]]) == 0
    assert torch.count_nonzero(removed['motor_drive'][0]) == 0
    torch.testing.assert_close(actual['motor_drive'][1], removed['motor_drive'][1])


def test_real_readout_normalization_matches_original_body_contract():
    torch.manual_seed(1)
    model = RegisteredPreview().eval()
    rng = np.random.default_rng(8)
    mean, scale = rng.normal(size=1024), rng.uniform(.5, 2, size=1024)
    body_mean, body_scale = rng.normal(size=21), rng.uniform(.5, 2, size=21)
    target_mean, target_scale = rng.normal(size=6), rng.uniform(.5, 2, size=6)
    adapter = bridge(model, feature_mean=mean, feature_scale=scale, body_mean=body_mean,
                     body_scale=body_scale, target_mean=target_mean, target_scale=target_scale)
    visual, body, valid, views, speed, goals = inputs(3)
    visual.copy_(torch.tensor(rng.normal(size=(3, 1024)), dtype=torch.float32))
    valid[0, 0] = False
    expected_body = body_features(body.numpy(), valid.numpy(), body_mean, body_scale, [0, 1, 4, 16])
    normalized = np.clip((visual.numpy() - mean) / scale, -5, 5).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.tensor(normalized), torch.tensor(expected_body))[:, 5]
        expected = -(expected * float(target_scale[5]) + float(target_mean[5])) * body[:, 16]
        actual = adapter(visual, body, valid, views, speed, goals)
    torch.testing.assert_close(actual['command'][:, 0], expected.clamp(-2, 2), atol=2e-6, rtol=2e-6)


def test_missing_view_or_speed_fails_and_yaw_is_bounded():
    adapter = bridge(MotionPreview())
    visual, body, valid, views, speed, goals = inputs(1)
    visual[:, 384] = 1000
    body[:, 16] = 100
    result = adapter(visual, body, valid, views, speed, goals)
    assert result['yaw_was_clipped'].item() and result['command'][0, 0].item() == -2
    views[0, 2] = False
    with pytest.raises(ValueError, match='Complete registered vision'):
        adapter(visual, body, valid, views, speed, goals)
    views[:] = True
    valid[0, 16] = False
    with pytest.raises(ValueError, match='Complete registered vision'):
        adapter(visual, body, valid, views, speed, goals)


def test_invalid_external_speed_ceiling_is_rejected_before_motor_drive():
    adapter = bridge(MotionPreview())
    visual, body, valid, views, speed, goals = inputs(1)
    speed[0] = 30
    with pytest.raises(ValueError, match='Complete registered vision'):
        adapter(visual, body, valid, views, speed, goals)
    speed[0] = 10
    goals[0, 0] = .5
    with pytest.raises(ValueError, match='Complete registered vision'):
        adapter(visual, body, valid, views, speed, goals)
