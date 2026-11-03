"""Research-only publication guard for a unified circuit; no actuator interface.

Host-clock freshness does not establish physical exposure or sensor sample time.
"""
import math
import time
from dataclasses import dataclass

import torch

from .temporal_body_context import HostFreshnessLimits, PerViewTemporalBodyContext
from .unified_flight_sensorimotor import UnifiedFlightSensorimotor


@dataclass(frozen=True)
class GuardedIntent:
    intent: torch.Tensor
    muscles: torch.Tensor
    decision_time: float
    completed_time: float
    physical_synchronization_verified: bool = False
    actor_qualified: bool = False


class UnifiedHostGuard:
    """Suppress priming and stale outputs, latch failures until explicit reset.

    This is an inference-only host timing guard. It is not vehicle authority,
    simulator integration, physical-clock registration, or a training runner.
    """

    def __init__(self, model, *, clock=time.perf_counter):
        if not isinstance(model, UnifiedFlightSensorimotor) or type(model.freshness_limits) is not HostFreshnessLimits:
            raise ValueError('Unified circuit with explicit host limits required')
        if not callable(clock):
            raise ValueError('Host monotonic clock required')
        self.model, self.clock, self.failed = model, clock, False

    def reset(self):
        self.model.reset_episode()
        self.failed = False

    def step(self, *, decision_time, racing_goals, scene_features, scene_valid, body_context):
        if self.failed:
            raise RuntimeError('Guard failed; explicit restart required')
        try:
            if not isinstance(body_context, PerViewTemporalBodyContext):
                raise ValueError('Original per-view context required')
            started = self.clock()
            if not math.isfinite(started) or started < decision_time:
                raise ValueError('Decision cannot be in the future of host start')
            held = self.model.previous_body_receipt
            priming = self.model.previous_time is None
            if not priming and held is None:
                raise ValueError('Previous held body receipt required')
            intent, muscles = self.model.sensory_decision(
                decision_time=decision_time, racing_goals=racing_goals,
                scene_features=scene_features, scene_valid=scene_valid, body_context=body_context)
            if scene_features.is_cuda:
                torch.cuda.synchronize(scene_features.device)
            completed = self.clock()
            if not math.isfinite(completed) or completed < started:
                raise ValueError('Monotonic completed host time required')
            limits = self.model.freshness_limits
            body_age = completed - body_context.decision_body.original_received_monotonic_seconds
            view_ages = [completed - x for x in body_context.render_start_lowers]
            if not 0 <= body_age <= limits.maximum_body_receipt_age_seconds:
                raise ValueError('Current body receipt stale at output')
            if held is not None and not 0 <= completed-held <= limits.maximum_body_receipt_age_seconds:
                raise ValueError('Held body receipt stale at output')
            if not all(0 <= age <= limits.maximum_view_render_age_seconds for age in view_ages):
                raise ValueError('View render bound stale at output')
            if priming:
                return None
            return GuardedIntent(intent=intent, muscles=muscles, decision_time=float(decision_time),
                                 completed_time=float(completed))
        except Exception:
            self.failed = True
            raise
