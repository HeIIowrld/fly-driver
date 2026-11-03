from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import (
    TemporalContextOpponentFlightStream,
    VisualBodyContextBoundary,
)
from flybrain_drive.temporal_body_context import (
    missing_temporal_body_context,
    temporal_body_context,
)
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def context(decision, yaw, *, yaw_valid=True):
    old = np.zeros(21)
    old[4] = yaw
    mask = np.ones(21, dtype=bool)
    mask[4] = yaw_valid
    prior = NavigationBodyObservation('a', 1, decision-.04, decision-.01, old, mask, 'x')
    current = NavigationBodyObservation('a', 2, decision-.005, decision,
                                        np.full(21, .25), np.ones(21, dtype=bool), 'x')
    return temporal_body_context(prior, current, render_start_lower=decision-.03,
                                 camera_receipt_time=decision-.01, decision_time=decision)


def test_context_projection_is_neutral_then_requires_yaw_and_motion():
    boundary = VisualBodyContextBoundary(1)
    fast = torch.zeros(1, 512)
    fast.reshape(1, 4, 2, 4, 16)[0, 0, 0, 1, 0] = 2
    traces = (fast, torch.zeros_like(fast), torch.zeros_like(fast))
    assert boundary(traces, context(1., .5)).item() == 0
    with torch.no_grad():
        boundary.projection.weight[0, 0] = 1
    assert boundary(traces, context(1., .5)).item() > 0
    assert boundary(traces, context(1., -.5)).item() < 0
    assert boundary(traces, context(1., .5, yaw_valid=False)).item() == 0
    assert boundary((fast, fast, fast), context(1., .5)).item() == 0
    missing = missing_temporal_body_context(context(1., .5).decision_body,
                                            decision_time=1., reason='receipt_too_old')
    assert boundary(traces, missing).item() == 0


def test_stream_keeps_current_body_and_rolls_back_after_invalid_context():
    seen = []

    class Joint:
        circuit = SimpleNamespace(backbone=SimpleNamespace(vision_ports=torch.tensor([0])),
                                  _read_state_tensor=lambda state: (state, state))

        def initial_state(self, batch):
            return torch.zeros(batch, 2)

        def drive_observed(self, visual, visual_valid, body, valid, *args):
            seen.append(body.clone())
            return torch.tensor([[.5, .2]])

        def advance(self, drive, state, elapsed):
            return None, state+drive*elapsed

    stream = TemporalContextOpponentFlightStream.__new__(TemporalContextOpponentFlightStream)
    stream.joint = Joint()
    stream.temporal = OpponentTemporalMotionBoundary(torch.zeros(576), torch.ones(576), 1)
    stream.context_boundary = VisualBodyContextBoundary(1)
    stream.reset_episode()
    scene = torch.zeros(1, 576)

    def step(t, body_context):
        return stream.decision(decision_time=t, visual=None, visual_valid=None,
            body_context=body_context, goals=torch.tensor([[0., 1.]]), scene_features=scene,
            scene_valid=torch.ones(1, 4, dtype=torch.bool), airflow_observation=None)[0]

    torch.testing.assert_close(step(1., context(1., .5)), torch.zeros(1, 2))
    torch.testing.assert_close(seen[0], torch.full((1, 21), .25))
    torch.testing.assert_close(step(1.04, context(1.04, -.5)), torch.tensor([[.02, .008]]))
    state, prior_time = stream.state.clone(), stream.previous_time
    bad = replace(context(1.05, .5), render_start_lower=1.0)
    with pytest.raises(ValueError):
        step(1.05, bad)
    torch.testing.assert_close(stream.state, state)
    assert stream.previous_time == prior_time
    stream.reset_episode()
    torch.testing.assert_close(step(2., context(2., .5)), torch.zeros(1, 2))
