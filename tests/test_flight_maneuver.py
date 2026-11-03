import numpy as np
import pytest
import torch

from flybrain_drive.flight_learning import GOAL_CONTRACT
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, ManeuverCurrents
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal


def invoke(encoder, command, body, valid, goals, **extra):
    return encoder(command, body, valid, goals, command_contract=MANEUVER_CONTRACT,
                   goal_contract=RACING_GOAL_CONTRACT, **extra)


def inputs():
    body = torch.zeros(3, 21)
    valid = torch.zeros_like(body, dtype=torch.bool)
    valid[:, [4, 16]] = True
    body[:, 16] = 60/3.6
    command = torch.tensor([[.3, 60/3.6], [-.3, 100/3.6], [.3, 30/3.6]])
    return command, body, valid, torch.tensor(racing_goal(100, 3))


def test_physical_units_high_speed_and_no_yaw_subtraction_at_command_port():
    encoder = ManeuverCurrents([-1, 1, -1, 1], 4)
    command, body, valid, goals = inputs()
    turn, power = invoke(encoder, command, body, valid, goals)
    assert power[1, 0] > power[0, 0] > power[2, 0]
    torch.testing.assert_close(power[0], torch.full((4,), .05))
    torch.testing.assert_close(turn[0], turn[2])
    assert turn[0, 1] > 0 and turn[0, 0] == 0
    assert turn[1, 0] > 0 and turn[1, 1] == 0
    body[:, 4] = .3
    changed_turn, changed_power = invoke(encoder, command, body, valid, goals)
    torch.testing.assert_close(turn, changed_turn, rtol=0, atol=0)
    torch.testing.assert_close(power, changed_power, rtol=0, atol=0)


def test_separate_path_interventions_and_ceiling_is_not_cruise_target():
    encoder = ManeuverCurrents([-1, 1, -1, 1], 2)
    command, body, valid, goals = inputs()
    a = invoke(encoder, command, body, valid, goals)
    higher_ceiling = torch.tensor(racing_goal(200, 3))
    b = invoke(encoder, command, body, valid, higher_ceiling)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    turn, power = invoke(encoder, command, body, valid, goals, lesion='desired_yaw')
    assert not turn.any()
    torch.testing.assert_close(power, a[1])
    turn, power = invoke(encoder, command, body, valid, goals, lesion='speed_error')
    torch.testing.assert_close(turn, a[0])
    torch.testing.assert_close(power, torch.full((3, 2), .05))


@pytest.mark.parametrize('fault', ['unknown_yaw', 'unknown_speed', 'above_ceiling', 'heading_bias',
                                 'negative_speed', 'nonfinite', 'wrong_shape', 'wrong_dtype'])
def test_invalid_or_legacy_inputs_are_not_silently_repaired(fault):
    encoder = ManeuverCurrents([-1, 1, -1, 1], 2)
    command, body, valid, goals = inputs()
    if fault == 'unknown_yaw':
        valid[0, 4] = False
    elif fault == 'unknown_speed':
        valid[0, 16] = False
    elif fault == 'above_ceiling':
        command[0, 1] = 101/3.6
    elif fault == 'heading_bias':
        goals[0, 0] = .1
    elif fault == 'negative_speed':
        body[0, 16] = -1
    elif fault == 'nonfinite':
        command[0, 0] = np.nan
    elif fault == 'wrong_shape':
        command = command[:, :1]
    else:
        body = body.double()
    with pytest.raises(ValueError):
        invoke(encoder, command, body, valid, goals)


def test_legacy_task_contract_is_rejected_even_with_matching_tensor_shapes():
    encoder = ManeuverCurrents([-1, 1, -1, 1], 2)
    command, body, valid, goals = inputs()
    with pytest.raises(ValueError):
        encoder(command, body, valid, goals, command_contract=GOAL_CONTRACT,
                goal_contract=RACING_GOAL_CONTRACT)
    with pytest.raises(ValueError):
        encoder(command, body, valid, goals, command_contract=MANEUVER_CONTRACT,
                goal_contract=GOAL_CONTRACT)
