"""Causal body-rate updates over held four-view flight features; offline research only."""

import math

import torch

from .flight_visual_body_temporal_context import PerViewTemporalContextOpponentFlightStream
from .relative_airflow_sensor import missing_airflow
from .temporal_body_context import PerViewTemporalBodyContext, per_view_temporal_body_context
from .virtual_navigation_body import NavigationBodyObservation


class MultirateFlightStream:
    """A new image or a body event advances one persistent original flight stream.

    The caller timestamps image readiness and native receipt in one host clock and
    must not supply either before it is actually available. Outputs are research
    intents only, never actuator commands. No source-time claim is made.
    """

    def __init__(self, stream, *, maximum_view_age=.15, maximum_body_age=.05):
        if (not isinstance(stream, PerViewTemporalContextOpponentFlightStream)
                or not 0 < maximum_view_age <= .15 or not 0 < maximum_body_age <= .05):
            raise ValueError('Explicit original flight stream and bounded receipt ages required')
        self.stream = stream
        self.maximum_view_age, self.maximum_body_age = maximum_view_age, maximum_body_age
        self.scene, self.visual_context = None, None
        self.previous_body_receipt, self.previous_time = None, None

    @torch.no_grad()
    def update(self, *, when, body, goals, visual, visual_valid, scene_valid,
               new_scene=None, new_visual_context=None):
        if (isinstance(when, bool) or not isinstance(when, (int, float))
                or not math.isfinite(when) or when < 0
                or self.previous_time is not None and when <= self.previous_time
                or not isinstance(body, NavigationBodyObservation)
                or body.decision_monotonic_seconds != when
                or not 0 <= when-body.original_received_monotonic_seconds <= self.maximum_body_age
                or not body.valid[[0, 1, 4, 16, 17, 18, 19, 20]].all()):
            raise ValueError('Advancing event and fresh original required body sample needed')
        if (new_scene is None) != (new_visual_context is None):
            raise ValueError('New scene and its original view context must arrive together')
        scene = self.scene if new_scene is None else new_scene
        frame = self.visual_context if new_visual_context is None else new_visual_context
        if (not isinstance(frame, PerViewTemporalBodyContext) or scene is None
                or frame.camera_receipt_time > when
                or max(when-lower for lower in frame.render_start_lowers) > self.maximum_view_age):
            raise ValueError('Available, nonstale original four-view scene required')
        context = per_view_temporal_body_context(frame.visual_contexts, body,
            render_start_lowers=frame.render_start_lowers,
            camera_receipt_time=frame.camera_receipt_time, decision_time=when)
        previous = self.previous_body_receipt
        intent, muscles = self.stream.decision(decision_time=when, visual=visual,
            visual_valid=visual_valid, body_context=context, goals=goals, scene_features=scene,
            scene_valid=scene_valid, airflow_observation=missing_airflow(when))
        self.scene, self.visual_context = scene.detach().clone(), frame
        self.previous_body_receipt = body.original_received_monotonic_seconds
        self.previous_time = float(when)
        return intent, muscles, {'decision_time': float(when),
            'new_body_receipt': body.original_received_monotonic_seconds,
            'previous_held_body_receipt': previous,
            'previous_held_body_age_at_decision': None if previous is None else when-previous,
            'visual_replaced': new_scene is not None, 'source_time_verified': False,
            'output_time_freshness_verified': False, 'vehicle_control': False}
