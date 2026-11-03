from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.flight_counterfactual import FlightCounterfactualReplay


class IntegratingFlight:
    spec = {'dynamics': {'dt': .01}}

    def initial_state(self, count):
        return torch.zeros(count, 1)

    def _advance_elapsed_tensor(self, held, state, dt):
        return state+held*dt

    def _read_state_tensor(self, state):
        return state.repeat(1, 2), None

    def _drive_tensor(self, visual, visual_valid, body, body_valid, goals, features,
                      *, lesion=None):
        scene = 0. if lesion == 'scene_command' else features.sum(1, keepdim=True)
        return body[:, :1]+2*body[:, 4:5]+scene


def stream():
    body = np.zeros(16, dtype=np.float32)
    body[0], body[4] = .1, .02
    return SimpleNamespace(yaw_trial_mode='full', flight=IntegratingFlight(),
        goals=torch.zeros(1, 2), visual_zero=torch.zeros(1, 2),
        visual_valid=torch.zeros(1, 2, dtype=torch.bool),
        last_features=torch.tensor([[.1, .2]]),
        last_body=SimpleNamespace(values=body, valid=np.ones(16, dtype=bool)))


def test_original_clock_prior_input_hold_fractional_tick_and_no_source_mutation():
    value = stream()
    original_body, original_valid = value.last_body.values.copy(), value.last_body.valid.copy()
    replay = FlightCounterfactualReplay(value)
    np.testing.assert_array_equal(replay.observe(value, 1., [0, 0]), np.zeros((5, 2)))
    np.testing.assert_array_equal(value.last_body.values, original_body)
    np.testing.assert_array_equal(value.last_body.valid, original_valid)
    # Today's input cannot change the output integrated over the preceding37ms.
    value.last_body.values[4] = -.05
    value.last_features[:] = .1
    actual = replay.observe(value, 1.037, [.44*.037]*2)
    np.testing.assert_allclose(actual[:, 0], np.array([.44, .4, .36, .14, .1])*.037,
                               atol=1e-8)
    actual = replay.observe(value, 1.073, [.44*.037+.2*.036]*2)
    expected = np.array([.44, .4, .36, .14, .1])*.037
    expected += np.array([.2, .3, .4, 0., .1])*.036
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-8)
    assert replay.maximum_error < 1e-7 and len(replay.rows) == 3
    assert value.last_body.values[4] == np.float32(-.05)


def test_unknown_drive_or_reset_clock_cannot_be_silently_accepted():
    value = stream()
    value.yaw_trial_mode = 'zero_yaw'
    with pytest.raises(ValueError, match='Unmodified'):
        FlightCounterfactualReplay(value)
    value.yaw_trial_mode = 'full'
    replay = FlightCounterfactualReplay(value)
    replay.observe(value, 1., [0, 0])
    with pytest.raises(ValueError, match='Original positive interval'):
        replay.observe(value, 1., [0, 0])
