import pytest
import torch

from flybrain_drive.flight_motion_forecast import CausalMotionForecast
from test_unified_flight_sensorimotor import context, make_model


def test_prediction_is_causal_and_preserves_intensity():
    forecast = CausalMotionForecast()
    with torch.no_grad():
        forecast.raw_gain.fill_(0.5)
    previous = torch.zeros(1, 576)
    current = previous.reshape(1, 4, 9, 16).clone()
    current[:, :, 0] = .3
    current[:, :, 1:] = .2
    current = current.reshape(1, 576)
    assert torch.equal(forecast(current, None, None), current)
    result = forecast(current, previous, .04).reshape(1, 4, 9, 16)
    torch.testing.assert_close(result[:, :, 0], torch.full((1, 4, 16), .3))
    torch.testing.assert_close(result[:, :, 1:], torch.full((1, 4, 8, 16), .2 * (1 + torch.tanh(torch.tensor(.5)))))
    with pytest.raises(ValueError, match='40ms'):
        forecast(current, previous, .01)


def test_integrated_forecast_resets_and_failed_tick_does_not_advance_state():
    model = make_model()
    model.motion_forecast = CausalMotionForecast()
    args = dict(racing_goals=torch.tensor([[0., 1.]]), scene_features=torch.zeros(1, 576),
                scene_valid=torch.ones(1, 4, dtype=torch.bool))
    model.sensory_decision(decision_time=1., body_context=context(1.), **args)
    previous = model.previous_raw_scene.clone()
    with pytest.raises(ValueError, match='40ms'):
        model.sensory_decision(decision_time=1.01, body_context=context(1.01), **args)
    assert model.previous_time == 1. and torch.equal(model.previous_raw_scene, previous)
    model.sensory_decision(decision_time=1.04, body_context=context(1.04), **args)
    model.reset_episode()
    assert model.previous_raw_scene is None and model.state is None


def test_zero_forecast_matches_original_unified_graph():
    baseline, candidate = make_model(), make_model()
    candidate.load_state_dict(baseline.state_dict())
    candidate.motion_forecast = CausalMotionForecast()
    for i in range(4):
        now = 1. + .04*i
        scene = torch.full((1, 576), i*.03)
        args = dict(decision_time=now, racing_goals=torch.tensor([[0., 1.]]),
                    scene_features=scene, scene_valid=torch.ones(1, 4, dtype=torch.bool))
        a = baseline.sensory_decision(body_context=context(now), **args)
        b = candidate.sensory_decision(body_context=context(now), **args)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert torch.equal(baseline.state, candidate.state)
