from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.flight_sensory_stream import FlightSensoryStream


def test_held_drive_state_and_rejection_do_not_reset_or_advance():
    class Joint:
        circuit = SimpleNamespace(_read_state_tensor=lambda state: (state, state + 1))

        def initial_state(self, batch):
            return torch.zeros((batch, 1))

        def drive_observed(self, *args):
            if args[-2][0] == 'invalid':
                raise ValueError('invalid sensor')
            return torch.tensor([[float(args[-2][0])]])

        def advance(self, drive, state, elapsed):
            return None, state + drive * elapsed

    stream = FlightSensoryStream.__new__(FlightSensoryStream)
    stream.joint = Joint()
    stream.reset_episode()

    def step(time, value, goals=None):
        return stream.decision(
            decision_time=time, visual=None, visual_valid=None, body=torch.zeros((1, 21)),
            body_valid=None, goals=torch.tensor([[0., 1.]]) if goals is None else goals,
            scene_features=None, scene_valid=None,
            airflow_observation=value,
        )[0]

    torch.testing.assert_close(step(1., 2.), torch.tensor([[0.]]))
    torch.testing.assert_close(step(1.05, 4.), torch.tensor([[.1]]))
    with pytest.raises(ValueError, match='invalid sensor'):
        step(1.06, 'invalid')
    with pytest.raises(ValueError, match='80ms'):
        step(1.14, 4.)
    with pytest.raises(ValueError, match='zero route bias'):
        step(1.06, 4., torch.tensor([[.2, 1.]]))
    with pytest.raises(ValueError, match='0..30km/h'):
        step(1.06, 4., torch.tensor([[0., 1.1]]))
    torch.testing.assert_close(step(1.07, 3.), torch.tensor([[.18]]))
    stream.reset_episode()
    torch.testing.assert_close(step(3., 3.), torch.tensor([[0.]]))
