import pytest
import torch

from flybrain_drive.flight_airflow_feedback import (
    ExperimentalAirflowBodyCircuit,
    RelativeAirflowAdapter,
)
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT
from tests.test_flight_body_feedback_adapter import FakeMotor


def test_missing_airflow_is_neutral_even_after_learning_and_values_are_not_fabricated():
    adapter = RelativeAirflowAdapter(2)
    with torch.no_grad():
        adapter.projection.weight.fill_(1.)
    invalid = torch.zeros((2, 3), dtype=torch.bool)
    actual = adapter(torch.tensor([[100., 2., 1.], [-100., -7., 10.]]), invalid)
    assert torch.equal(actual, torch.zeros_like(actual))
    valid = invalid.clone()
    valid[:, 1] = True
    value = adapter(torch.tensor([[100., 2., 1.], [-100., 2., 10.]]), valid)
    assert torch.equal(value[0], value[1]) and torch.any(value != 0)
    with pytest.raises(ValueError, match='mask'):
        adapter(torch.zeros(1, 3), torch.ones(1, 3))


def test_airflow_modulates_only_body_ports_and_graph_receives_gradient():
    circuit = ExperimentalAirflowBodyCircuit(FakeMotor())
    body, valid = torch.zeros(1, 21), torch.ones(1, 21, dtype=torch.bool)
    flow, mask = torch.tensor([[3., 5., 0.]]), torch.ones(1, 3, dtype=torch.bool)
    command, goals = torch.zeros(1, 2), torch.ones(1, 2)
    kw = {'command_contract': MANEUVER_CONTRACT, 'goal_contract': RACING_GOAL_CONTRACT}
    base = circuit.motor.drive(command, body, valid, goals, **kw)
    assert torch.equal(circuit.drive(command, body, valid, goals, flow, mask, **kw), base)
    _, _, state = circuit.advance(command, body, valid, goals, flow, mask,
                                   circuit.initial_state(1), .025, **kw)
    state.sum().backward()
    assert circuit.airflow_adapter.projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        circuit.airflow_adapter.projection.weight.fill_(1.)
    changed = circuit.drive(command, body, valid, goals, flow, mask, **kw)
    assert torch.equal(changed[:, [0, 2, 4]], base[:, [0, 2, 4]])
    assert torch.any(changed[:, [1, 3]] != base[:, [1, 3]])
