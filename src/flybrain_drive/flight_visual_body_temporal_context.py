"""Experimental historical body × visual-motion boundary; not measured fly wiring."""

import math

import torch
from torch import nn

from .flight_opponent_sensory_stream import OpponentFlightSensoryStream
from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from .flyvis_camera import integration_intervals
from .packed_view_body import pack_view_body
from .temporal_body_context import (
    MissingTemporalBodyContext,
    PerViewTemporalBodyContext,
    TemporalBodyContext,
    missing_temporal_body_context,
    per_view_temporal_body_context,
    temporal_body_context,
)


class VisualBodyContextBoundary(nn.Module):
    """Neutral projection of masked historical yaw and four-view opponent traces."""

    def __init__(self, port_count, *, device=None, dtype=torch.float32):
        super().__init__()
        if type(port_count) is not int or port_count < 1:
            raise ValueError('Explicit visual port count required')
        self.projection = nn.Linear(256, port_count, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.projection.weight)

    def forward(self, traces, context):
        if (not isinstance(context, (TemporalBodyContext, MissingTemporalBodyContext))
                or not isinstance(traces, tuple) or len(traces) != 3
                or any(not isinstance(t, torch.Tensor) or t.shape != (1, 512)
                       or t.device != self.projection.weight.device
                       or t.dtype != self.projection.weight.dtype or not torch.isfinite(t).all()
                       for t in traces)):
            raise ValueError('One ordered temporal trace and explicit body context required')
        if isinstance(context, MissingTemporalBodyContext):
            return self.projection.weight.new_zeros((1, self.projection.out_features))
        history = context.visual_context
        if history.values.shape != (21,) or history.valid.shape != (21,):
            raise ValueError('Original 21-channel historical body and mask required')
        yaw = float(history.values[4]) if history.valid[4] else 0.
        if not math.isfinite(yaw):
            raise ValueError('Finite historical yaw required')
        motion = (traces[0]-traces[1]).reshape(1, 4, 2, 4, 16)
        opponent = torch.stack((motion[:, :, :, 1]-motion[:, :, :, 0],
                                motion[:, :, :, 2]-motion[:, :, :, 3]), dim=3)
        # 1 rad/s is a numerical scaling assumption, not a measured fly tuning curve.
        interaction = opponent.reshape(1, 256) * max(-1., min(1., yaw))
        return .5 * torch.tanh(self.projection(interaction))


class PerViewVisualBodyContextBoundary(nn.Module):
    """Per-view host-prior yaw × opponent traces; engineering approximation only.

    The caller must establish receipt order and retain the individual view masks.
    This boundary neither establishes physical exposure timing nor generates controls.
    """

    def __init__(self, port_count, *, device=None, dtype=torch.float32):
        super().__init__()
        if type(port_count) is not int or port_count < 1:
            raise ValueError('Explicit visual port count required')
        self.projection = nn.Linear(256, port_count, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.projection.weight)

    def forward(self, traces, body, valid):
        if (not isinstance(traces, tuple) or len(traces) != 3
                or any(not isinstance(t, torch.Tensor) or t.shape != (1, 512)
                       or t.device != self.projection.weight.device
                       or t.dtype != self.projection.weight.dtype or not torch.isfinite(t).all()
                       for t in traces)
                or not isinstance(body, torch.Tensor) or body.shape != (1, 4, 21)
                or body.device != self.projection.weight.device
                or body.dtype != self.projection.weight.dtype or not torch.isfinite(body).all()
                or not isinstance(valid, torch.Tensor) or valid.shape != body.shape
                or valid.dtype != torch.bool or valid.device != body.device):
            raise ValueError('Four explicit finite body values/masks and temporal traces required')
        motion = (traces[0]-traces[1]).reshape(1, 4, 2, 4, 16)
        opponent = torch.stack((motion[:, :, :, 1]-motion[:, :, :, 0],
                                motion[:, :, :, 2]-motion[:, :, :, 3]), dim=3)
        yaw = torch.where(valid[:, :, 4], body[:, :, 4].clamp(-1, 1),
                          torch.zeros_like(body[:, :, 4]))
        interaction = (opponent * yaw[:, :, None, None, None]).reshape(1, 256)
        return .5*torch.tanh(self.projection(interaction))


class TemporalContextOpponentFlightStream(OpponentFlightSensoryStream):
    """Keep decision body current; use earlier body only on HS/VS visual ports."""

    def __init__(self, joint, temporal, context_boundary):
        super().__init__(joint, temporal)
        if (not isinstance(context_boundary, VisualBodyContextBoundary)
                or context_boundary.projection.out_features != len(
                    joint.circuit.backbone.vision_ports)):
            raise ValueError('Matching explicit historical visual boundary required')
        self.context_boundary = context_boundary

    @torch.no_grad()
    def decision(self, *, decision_time, visual, visual_valid, body_context, goals,
                 scene_features, scene_valid, airflow_observation):
        if (not isinstance(body_context, (TemporalBodyContext, MissingTemporalBodyContext))
                or body_context.decision_time != decision_time):
            raise ValueError('Exact original decision-time body context required')
        if isinstance(body_context, MissingTemporalBodyContext):
            missing_temporal_body_context(body_context.decision_body,
                                         decision_time=decision_time, reason=body_context.reason)
        else:
            temporal_body_context(body_context.visual_context, body_context.decision_body,
                camera_receipt_time=body_context.camera_receipt_time,
                render_start_lower=body_context.render_start_lower, decision_time=decision_time)
        elapsed = None if self.previous_time is None else decision_time-self.previous_time
        if elapsed is not None and not 0 < elapsed <= .08:
            raise ValueError('Original advancing decision interval up to 80ms required')
        if (not isinstance(goals, torch.Tensor) or goals.shape != (1, 2)
                or not torch.isfinite(goals).all() or goals[0, 0] != 0
                or not 0 <= goals[0, 1] <= 1):
            raise ValueError('Declared zero route bias and 0..30km/h task required')
        current = body_context.decision_body
        body = torch.tensor(current.values.copy(), dtype=scene_features.dtype,
                            device=scene_features.device).unsqueeze(0)
        valid = torch.tensor(current.valid.copy(), dtype=torch.bool,
                             device=scene_features.device).unsqueeze(0)
        drive = self.joint.drive_observed(visual, visual_valid, body, valid, goals,
            scene_features, scene_valid, [airflow_observation], [decision_time])
        delta, traces = self.temporal.step(scene_features, scene_valid, self.traces, elapsed)
        ports = self.joint.circuit.backbone.vision_ports
        drive = self.temporal.modulate(drive, ports, delta)
        drive = self.temporal.modulate(drive, ports, self.context_boundary(traces, body_context))
        state = self.state
        if elapsed is not None:
            _, state = self.joint.advance(self.previous_drive, self.state, elapsed)
        intent, muscles = self.joint.circuit._read_state_tensor(state)
        self.state = state.detach().clone()
        self.traces = tuple(value.detach().clone() for value in traces)
        self.previous_drive = drive.detach().clone()
        self.previous_time = float(decision_time)
        return intent.detach().clone(), muscles.detach().clone()


class PerViewTemporalContextOpponentFlightStream(OpponentFlightSensoryStream):
    """Experimental four-view historical and decision-body boundary; no actuator."""

    def __init__(self, joint, temporal, context_boundary, *, optimize_held=False):
        super().__init__(joint, temporal)
        if (not isinstance(context_boundary, PerViewVisualBodyContextBoundary)
                or context_boundary.projection.out_features != len(
                    joint.circuit.backbone.vision_ports)):
            raise ValueError('Matching explicit per-view visual boundary required')
        self.context_boundary = context_boundary
        if type(optimize_held) is not bool:
            raise ValueError('Explicit held-step optimization setting required')
        if optimize_held and (type(temporal) is not OpponentTemporalMotionBoundary
                or type(joint) is not ExperimentalVisualBodyAirflow
                or joint.circuit.backbone.spec['dynamics']['dt'] < .005):
            raise ValueError('Exact original temporal and joint 5ms-compatible integrators required')
        self.optimize_held = optimize_held

    def reset_episode(self):
        super().reset_episode()
        self.previous_scene = None

    @torch.no_grad()
    def decision(self, *, decision_time, visual, visual_valid, body_context, goals,
                 scene_features, scene_valid, airflow_observation):
        if (not isinstance(body_context, PerViewTemporalBodyContext)
                or body_context.decision_time != decision_time):
            raise ValueError('Exact decision-time per-view body context required')
        per_view_temporal_body_context(body_context.visual_contexts, body_context.decision_body,
            render_start_lowers=body_context.render_start_lowers,
            camera_receipt_time=body_context.camera_receipt_time, decision_time=decision_time)
        elapsed = None if self.previous_time is None else decision_time-self.previous_time
        if elapsed is not None and not 0 < elapsed <= .15:
            raise ValueError('Original advancing decision interval up to 150ms required')
        if (not isinstance(goals, torch.Tensor) or goals.shape != (1, 2)
                or not torch.isfinite(goals).all() or goals[0, 0] != 0
                or not 0 <= goals[0, 1] <= 1):
            raise ValueError('Declared zero route bias and 0..30km/h task required')
        current = body_context.decision_body
        body = torch.tensor(current.values.copy(), dtype=scene_features.dtype,
                            device=scene_features.device).unsqueeze(0)
        valid = torch.tensor(current.valid.copy(), dtype=torch.bool,
                             device=scene_features.device).unsqueeze(0)
        history, history_valid = pack_view_body(body_context.visual_contexts,
            device=scene_features.device, dtype=scene_features.dtype)
        drive = self.joint.drive_observed(visual, visual_valid, body, valid, goals,
            scene_features, scene_valid, [airflow_observation], [decision_time])
        state, traces = self.state, self.traces
        if elapsed is None:
            delta, traces = self.temporal.step(scene_features, scene_valid, None, None)
        else:
            intervals = integration_intervals(elapsed)
            if getattr(self, 'optimize_held', False) and len(intervals) > 1:
                # First and last steps retain the full public validators. Intermediate
                # steps see the identical held observation and drive; the original
                # exponential updates and graph substeps are not merged or retimed.
                _, traces = self.temporal.step(self.previous_scene, scene_valid, traces, intervals[0])
                _, state = self.joint.advance(self.previous_drive, state, intervals[0])
                for dt in intervals[1:-1]:
                    fast_alpha = math.exp(-dt/self.temporal.fast_seconds)
                    slow_alpha = math.exp(-dt/self.temporal.slow_seconds)
                    fast = fast_alpha*traces[0]+(1-fast_alpha)*traces[2]
                    slow = slow_alpha*traces[1]+(1-slow_alpha)*traces[2]
                    traces = (fast, slow, traces[2])
                    state = self.joint.circuit._advance_elapsed_tensor(self.previous_drive, state, dt)
                delta, traces = self.temporal.step(scene_features, scene_valid, traces, intervals[-1])
                _, state = self.joint.advance(self.previous_drive, state, intervals[-1])
            else:
                for index, dt in enumerate(intervals):
                    current = scene_features if index == len(intervals)-1 else self.previous_scene
                    delta, traces = self.temporal.step(current, scene_valid, traces, dt)
                    _, state = self.joint.advance(self.previous_drive, state, dt)
        ports = self.joint.circuit.backbone.vision_ports
        drive = self.temporal.modulate(drive, ports, delta)
        drive = self.temporal.modulate(drive, ports,
                                      self.context_boundary(traces, history, history_valid))
        intent, muscles = self.joint.circuit._read_state_tensor(state)
        self.state = state.detach().clone()
        self.traces = tuple(value.detach().clone() for value in traces)
        self.previous_drive = drive.detach().clone()
        self.previous_scene = scene_features.detach().clone()
        self.previous_time = float(decision_time)
        return intent.detach().clone(), muscles.detach().clone()
