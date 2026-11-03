import pytest
import torch

from flybrain_drive.flight_learning import step_connected_loss
from flybrain_drive.flight_navigation import SceneCommand


def test_scene_normalization_is_required_and_gate_has_nonzero_learning_gradient():
    command = SceneCommand(576, 4, baseline_current=.003)
    features = torch.ones(2, 576)
    with pytest.raises(ValueError, match="normalization"):
        command(features)
    command.set_train_normalization(torch.zeros(576), torch.ones(576))
    current = command(features)
    torch.testing.assert_close(current, torch.full((2, 4), .003))
    current.sum().backward()
    assert command.linear.weight.grad.abs().min() > 0


def test_visual_information_lesion_preserves_learned_bias_and_channel_identity():
    command = SceneCommand(576, 4, baseline_current=.003)
    command.set_train_normalization(torch.zeros(576), torch.ones(576))
    with torch.no_grad():
        command.linear.weight.fill_(.001)
    scene = torch.ones(2, 576)
    removed = command(scene, lesion="scene_command")
    torch.testing.assert_close(removed, command.linear.bias.sigmoid()[None].repeat(2, 1))
    assert (command(scene) > command(scene, lesion="motion_channels")).all()
    assert (command(scene, lesion="motion_channels") > removed).all()
    assert (command(scene, lesion="intensity_channel") > removed).all()


def test_detached_past_target_does_not_advance_adam_momentum():
    parameter = torch.nn.Parameter(torch.tensor(.5))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    assert step_connected_loss(parameter.square(), optimizer, [parameter])
    optimizer.zero_grad(set_to_none=True)
    held_past = parameter.detach().clone()
    value, step = parameter.detach().clone(), optimizer.state[parameter]["step"].clone()
    assert not step_connected_loss((held_past - 1).square(), optimizer, [parameter])
    torch.testing.assert_close(parameter, value, atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state[parameter]["step"], step, atol=0, rtol=0)
    assert step_connected_loss((parameter - 1).square(), optimizer, [parameter])
    assert optimizer.state[parameter]["step"] == step + 1
