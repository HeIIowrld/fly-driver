from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.flight_visual_body_temporal_context import (
    PerViewTemporalContextOpponentFlightStream,
    PerViewVisualBodyContextBoundary,
)
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def obs(packet, receipt, evaluation, yaw, *, episode='run'):
    values, valid = np.zeros(21, np.float32), np.ones(21, bool)
    values[4] = yaw
    return NavigationBodyObservation(episode, packet, receipt, evaluation, values, valid, 'hash')


def context(now=10.):
    camera = now-.01
    bounds = tuple(now-x for x in (.08, .07, .06, .05))
    views = (obs(1, now-.09, camera, .1), None,
             obs(3, now-.065, camera, .3), obs(4, now-.055, camera, -.2))
    current = obs(5, now-.005, now, .8)
    return per_view_temporal_body_context(views, current, render_start_lowers=bounds,
                                          camera_receipt_time=camera, decision_time=now)


def test_per_view_validates_each_order_and_provenance_even_when_one_is_missing():
    c = context()
    assert c.visual_contexts[1] is None and c.decision_body.values[4] == np.float32(.8)
    late = list(c.visual_contexts)
    late[0] = replace(late[0], original_received_monotonic_seconds=c.render_start_lowers[0])
    with pytest.raises(ValueError):
        per_view_temporal_body_context(tuple(late), c.decision_body,
            render_start_lowers=c.render_start_lowers, camera_receipt_time=c.camera_receipt_time,
            decision_time=c.decision_time)
    different = list(c.visual_contexts)
    different[2] = replace(different[2], episode_id='other')
    with pytest.raises(ValueError):
        per_view_temporal_body_context(tuple(different), c.decision_body,
            render_start_lowers=c.render_start_lowers, camera_receipt_time=c.camera_receipt_time,
            decision_time=c.decision_time)


def test_per_view_stream_keeps_current_body_distinct_and_carries_previous_drive():
    class Temporal:
        def __init__(self):
            self.scenes = []

        def step(self, scene, valid, state, elapsed):
            self.scenes.append(scene.clone())
            t = torch.arange(512, dtype=torch.float32).reshape(1, 512)/512
            return torch.zeros(1, 1), (t, torch.zeros_like(t), t)

        def modulate(self, drive, ports, delta):
            return drive+delta

    class Joint:
        def __init__(self):
            self.circuit = SimpleNamespace(backbone=SimpleNamespace(vision_ports=torch.tensor([0])),
                _read_state_tensor=lambda state: (state, state))
            self.body_seen, self.held_seen = [], []

        def drive_observed(self, visual, visual_valid, body, valid, *args):
            self.body_seen.append(body.clone())
            return body[:, 4:5].clone()

        def advance(self, drive, state, elapsed):
            self.held_seen.append(drive.clone())
            return None, state+drive*elapsed

    stream = PerViewTemporalContextOpponentFlightStream.__new__(PerViewTemporalContextOpponentFlightStream)
    stream.joint, stream.temporal = Joint(), Temporal()
    stream.context_boundary = PerViewVisualBodyContextBoundary(1)
    stream.state, stream.traces, stream.previous_drive, stream.previous_time = torch.zeros(1, 1), None, None, None
    kwargs = dict(visual=None, visual_valid=None, goals=torch.tensor([[0., .5]]),
                  scene_features=torch.zeros(1, 576), scene_valid=None, airflow_observation=None)
    stream.decision(decision_time=10., body_context=context(), **kwargs)
    prior = stream.previous_drive.clone()
    kwargs['scene_features'] = torch.ones(1, 576)
    stream.decision(decision_time=10.1, body_context=context(10.1), **kwargs)
    torch.testing.assert_close(stream.joint.body_seen[0], torch.as_tensor(context().decision_body.values)[None])
    torch.testing.assert_close(stream.joint.held_seen[0], prior)
    torch.testing.assert_close(stream.state, prior*.1)
    assert all(torch.count_nonzero(x) == 0 for x in stream.temporal.scenes[:-1])
    assert torch.all(stream.temporal.scenes[-1] == 1)
    previous_state = stream.state.clone()
    with pytest.raises(ValueError):
        stream.decision(decision_time=10.1, body_context=context(10.1), **kwargs)
    torch.testing.assert_close(stream.state, previous_state)


def test_optimized_held_steps_match_every_reference_state_and_trace():
    class Joint:
        def __init__(self):
            self.circuit = SimpleNamespace(backbone=SimpleNamespace(vision_ports=torch.tensor([0])),
                _read_state_tensor=lambda state: (state, state),
                _advance_elapsed_tensor=lambda drive, state, dt: state + drive*dt)

        def drive_observed(self, visual, visual_valid, body, valid, *args):
            return body[:, 4:5].clone()

        def advance(self, drive, state, elapsed):
            return None, self.circuit._advance_elapsed_tensor(drive, state, elapsed)

    joint = Joint()
    temporal = OpponentTemporalMotionBoundary(torch.zeros(576), torch.ones(576), 1)
    boundary = PerViewVisualBodyContextBoundary(1)
    with torch.no_grad():
        temporal.opponent_projection.weight.fill_(.01)
        boundary.projection.weight.fill_(.001)
    pair = []
    for fast in (False, True):
        stream = PerViewTemporalContextOpponentFlightStream.__new__(PerViewTemporalContextOpponentFlightStream)
        stream.joint, stream.temporal, stream.context_boundary = joint, temporal, boundary
        stream.optimize_held = fast
        stream.state, stream.traces, stream.previous_drive, stream.previous_time = torch.zeros(1, 1), None, None, None
        stream.previous_scene = None
        pair.append(stream)
    generator = torch.Generator().manual_seed(123)
    for now in (10., 10.003, 10.103, 10.251):
        scene = torch.randn(1, 576, generator=generator)
        kwargs = dict(decision_time=now, body_context=context(now), visual=None, visual_valid=None,
            goals=torch.tensor([[0., .5]]), scene_features=scene,
            scene_valid=torch.ones(1, 4, dtype=torch.bool), airflow_observation=None)
        a, _ = pair[0].decision(**kwargs)
        b, _ = pair[1].decision(**kwargs)
        assert torch.equal(a, b)
        assert torch.equal(pair[0].state, pair[1].state)
        assert torch.equal(pair[0].previous_drive, pair[1].previous_drive)
        assert all(torch.equal(x, y) for x, y in zip(pair[0].traces, pair[1].traces, strict=True))
