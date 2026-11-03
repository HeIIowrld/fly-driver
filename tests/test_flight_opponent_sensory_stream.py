from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.flight_opponent_sensory_stream import OpponentFlightSensoryStream
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary


def test_joint_stream_holds_both_drives_and_rolls_back_invalid_observation():
    class Joint:
        circuit = SimpleNamespace(backbone=SimpleNamespace(vision_ports=torch.tensor([0])),
                                  _read_state_tensor=lambda state: (state, state))

        def initial_state(self, batch):
            return torch.zeros(batch, 2)

        def drive_observed(self, *args):
            airflow = args[-2][0]
            if airflow == 'missing_contract':
                raise ValueError('invalid airflow')
            return torch.tensor([[.5, float(airflow)]])

        def advance(self, drive, state, elapsed):
            return None, state + drive * elapsed

    stream = OpponentFlightSensoryStream.__new__(OpponentFlightSensoryStream)
    stream.joint = Joint()
    stream.temporal = OpponentTemporalMotionBoundary(torch.zeros(576), torch.ones(576), 1)
    with torch.no_grad():
        stream.temporal.opponent_projection.weight[0, 0] = 1
    stream.reset_episode()
    scene = torch.zeros(1, 4, 9, 16)

    def step(time, airflow, scene_value):
        return stream.decision(decision_time=time, visual=None, visual_valid=None,
            body=torch.zeros(1, 21), body_valid=None, goals=torch.tensor([[0., 1.]]),
            scene_features=scene_value.flatten(1), scene_valid=torch.ones(1, 4, dtype=torch.bool),
            airflow_observation=airflow)[0]

    torch.testing.assert_close(step(1., .2, scene), torch.zeros(1, 2))
    scene[0, 0, 2, 0] = 1
    torch.testing.assert_close(step(1.05, .4, scene), torch.tensor([[.025, .01]]))
    prev_time, prev_state = stream.previous_time, stream.state.clone()
    prev_traces = tuple(v.clone() for v in stream.traces)
    with pytest.raises(ValueError, match='invalid airflow'):
        step(1.06, 'missing_contract', scene)
    assert stream.previous_time == prev_time
    torch.testing.assert_close(stream.state, prev_state)
    for before, after in zip(prev_traces, stream.traces, strict=True):
        torch.testing.assert_close(before, after)
    state = step(1.07, .7, scene)
    torch.testing.assert_close(state, torch.tensor([[.035, .018]]))
    assert stream.previous_drive[0, 0] > .5
    stream.reset_episode()
    torch.testing.assert_close(step(3., .3, scene), torch.zeros(1, 2))
    assert stream.traces is not None and torch.equal(stream.traces[0], stream.traces[1])
