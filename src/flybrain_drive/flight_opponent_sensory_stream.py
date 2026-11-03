"""Offline joint visual, opponent-time, vehicle body and relative-airflow stream.

This research boundary does not synchronize simulator sensors, actuate a vehicle,
or qualify an unverified airflow source. Recurrent state resets only per episode.
"""

import math

import torch

from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow


class OpponentFlightSensoryStream:
    def __init__(self, joint, temporal):
        if (not isinstance(joint, ExperimentalVisualBodyAirflow)
                or not isinstance(temporal, OpponentTemporalMotionBoundary)):
            raise ValueError('Exact joint and opponent-time research boundaries required')
        backbone = joint.circuit.backbone
        if (len(backbone.vision_ports) != temporal.opponent_projection.out_features
                or torch.isin(backbone.vision_ports, backbone.body_ports).any()
                or not torch.equal(temporal.mean, backbone.scene_command.mean)
                or not torch.equal(temporal.scale, backbone.scene_command.scale)):
            raise ValueError('Matching visual normalization and disjoint body ports required')
        self.joint, self.temporal = joint, temporal
        self.reset_episode()

    def reset_episode(self):
        self.state = self.joint.initial_state(1)
        self.traces = None
        self.previous_drive = None
        self.previous_time = None

    @torch.no_grad()
    def decision(self, *, decision_time, visual, visual_valid, body, body_valid, goals,
                 scene_features, scene_valid, airflow_observation):
        if (isinstance(decision_time, bool) or not isinstance(decision_time, (int, float))
                or not math.isfinite(decision_time) or decision_time < 0):
            raise ValueError('Finite nonnegative decision time required')
        elapsed = None if self.previous_time is None else decision_time - self.previous_time
        if (not isinstance(body, torch.Tensor) or body.ndim != 2 or len(body) != 1
                or elapsed is not None and not 0 < elapsed <= .08):
            raise ValueError('One body sample and increasing intervals up to 80ms required')
        if (not isinstance(goals, torch.Tensor) or goals.shape != (1, 2)
                or not torch.isfinite(goals).all() or goals[0, 0] != 0
                or not 0 <= goals[0, 1] <= 1):
            raise ValueError('Only zero route bias and a declared 0..30km/h task are allowed')
        # Validate current inputs before committing either sensory or neuronal state.
        drive = self.joint.drive_observed(
            visual, visual_valid, body, body_valid, goals, scene_features, scene_valid,
            [airflow_observation], [decision_time],
        )
        delta, next_traces = self.temporal.step(scene_features, scene_valid, self.traces, elapsed)
        drive = self.temporal.modulate(drive, self.joint.circuit.backbone.vision_ports, delta)
        next_state = self.state
        if elapsed is not None:
            _, next_state = self.joint.advance(self.previous_drive, self.state, elapsed)
        intent, muscles = self.joint.circuit._read_state_tensor(next_state)
        self.state = next_state.detach().clone()
        self.traces = tuple(value.detach().clone() for value in next_traces)
        self.previous_drive = drive.detach().clone()
        self.previous_time = float(decision_time)
        return intent.detach().clone(), muscles.detach().clone()
