"""Stateful offline integration of joint vision, body and relative airflow.

This is not an actuator interface, live sensor synchronizer, or qualification for
vehicle control. Caller-supplied feature and body provenance remains mandatory.
"""

import math

import torch

from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow


class FlightSensoryStream:
    """Hold the last sensory drive over each original interval without state resets."""

    def __init__(self, joint):
        if not isinstance(joint, ExperimentalVisualBodyAirflow):
            raise ValueError('Exact experimental joint circuit required')
        self.joint = joint
        self.reset_episode()

    def reset_episode(self):
        """Explicit episode or run restart; never call at a gradient detach."""
        self.state = self.joint.initial_state(1)
        self.previous_drive = None
        self.previous_time = None

    @torch.no_grad()
    def decision(self, *, decision_time, visual, visual_valid, body, body_valid, goals,
                 scene_features, scene_valid, airflow_observation):
        if (isinstance(decision_time, bool) or not isinstance(decision_time, (int, float))
                or not math.isfinite(decision_time) or decision_time < 0):
            raise ValueError('Finite nonnegative decision time required')
        if (body.ndim != 2 or len(body) != 1
                or self.previous_time is not None
                and not 0 < decision_time - self.previous_time <= .08):
            raise ValueError('One body sample and increasing intervals up to 80ms required')
        if (not isinstance(goals, torch.Tensor) or goals.shape != (1, 2)
                or not torch.isfinite(goals).all() or goals[0, 0] != 0
                or not 0 <= goals[0, 1] <= 1):
            raise ValueError('Only zero route bias and a declared 0..30km/h task are allowed')
        # Validate current sensors before committing any recurrent state transition.
        drive = self.joint.drive_observed(
            visual, visual_valid, body, body_valid, goals, scene_features, scene_valid,
            [airflow_observation], [decision_time],
        )
        next_state = self.state
        if self.previous_time is not None:
            _, next_state = self.joint.advance(
                self.previous_drive, self.state, decision_time - self.previous_time
            )
        intent, muscles = self.joint.circuit._read_state_tensor(next_state)
        self.state = next_state
        self.previous_drive = drive.detach().clone()
        self.previous_time = float(decision_time)
        return intent.detach().clone(), muscles.detach().clone()
