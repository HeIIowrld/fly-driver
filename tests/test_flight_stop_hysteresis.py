import pytest
import torch

from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT
from flybrain_drive.flight_stop_hysteresis import (
    ExperimentalHystereticStopCircuit,
    StopGoalHysteresis,
)
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT
from tests.test_flight_stop_task_adapter import FakeMotor


def test_stop_state_holds_through_jitter_and_releases_at_declared_threshold():
    state = StopGoalHysteresis()
    assert [state.update(x) for x in (0., .05, 0., .05, .1, .05, .01)] == [1, 1, 1, 1, 0, 0, 1]
    state.reset()
    assert state.update(.05) == 0


def test_invalid_goals_do_not_mutate_stop_state():
    state = StopGoalHysteresis()
    assert state.update(0.) == 1
    for value in (-1., float('nan'), True):
        with pytest.raises(ValueError):
            state.update(value)
    assert state.update(.05) == 1


def test_gate_changes_only_declared_ports_and_rejects_nonbinary_values():
    circuit = ExperimentalHystereticStopCircuit(FakeMotor())
    body, valid = torch.zeros(2, 21), torch.ones(2, 21, dtype=torch.bool)
    command, goals = torch.zeros(2, 2), torch.ones(2, 2)
    kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
    with torch.no_grad():
        circuit.adapter.stop_delta.fill_(.5)
    base = circuit.motor.drive(command, body, valid, goals, **kwargs)
    changed = circuit.drive(command, body, valid, goals, torch.tensor([0., 1.]), **kwargs)
    assert torch.equal(changed[0], base[0])
    assert torch.equal(changed[1, [0, 2, 4]], base[1, [0, 2, 4]])
    assert torch.all(changed[1, [1, 3]] > base[1, [1, 3]])
    with pytest.raises(ValueError):
        circuit.drive(command, body, valid, goals, torch.tensor([0., .5]), **kwargs)
