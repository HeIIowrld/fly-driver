"""Explicit camera-receipt context and decision-time body clocks; no source-time claim."""

import math
from dataclasses import dataclass

import numpy as np

from .virtual_navigation_body import NavigationBodyObservation


@dataclass(frozen=True)
class TemporalBodyContext:
    """Two observations at their actual host evaluation times, not synchronized exposure."""

    visual_context: NavigationBodyObservation
    decision_body: NavigationBodyObservation
    camera_receipt_time: float
    render_start_lower: float
    decision_time: float
    source_synchronization_verified: bool = False


@dataclass(frozen=True)
class MissingTemporalBodyContext:
    """Current body remains available; missing historical context is never imputed."""

    decision_body: NavigationBodyObservation
    decision_time: float
    reason: str


@dataclass(frozen=True)
class PerViewTemporalBodyContext:
    """Four separately sourced historical observations plus one current observation.

    Entries follow front, left, right, rear; None is explicit missing history.
    Host receipt ordering does not establish physical sample or exposure timing.
    """

    visual_contexts: tuple
    decision_body: NavigationBodyObservation
    render_start_lowers: tuple
    camera_receipt_time: float
    decision_time: float
    source_synchronization_verified: bool = False


def per_view_temporal_body_context(visual_contexts, decision_body, *, render_start_lowers,
                                   camera_receipt_time, decision_time):
    if (not isinstance(visual_contexts, tuple) or len(visual_contexts) != 4
            or not isinstance(render_start_lowers, tuple) or len(render_start_lowers) != 4):
        raise ValueError('Four ordered view contexts and render bounds required')
    # Also validate current body and clocks when all four histories are missing.
    missing_temporal_body_context(decision_body, decision_time=decision_time,
                                  reason='unavailable_historical_receipt')
    for observation in (decision_body, *visual_contexts):
        if observation is not None and (
                not isinstance(observation, NavigationBodyObservation)
                or not isinstance(observation.values, np.ndarray)
                or observation.values.shape != (21,) or not np.isfinite(observation.values).all()
                or not isinstance(observation.valid, np.ndarray)
                or observation.valid.shape != (21,) or observation.valid.dtype != np.bool_):
            raise ValueError('Original finite body values and boolean masks required')
    for context, lower in zip(visual_contexts, render_start_lowers, strict=True):
        if (isinstance(lower, bool) or not isinstance(lower, (int, float))
                or not math.isfinite(lower) or lower < 0
                or not isinstance(camera_receipt_time, (int, float))
                or isinstance(camera_receipt_time, bool)
                or not math.isfinite(camera_receipt_time)
                or not lower <= camera_receipt_time <= decision_time):
            raise ValueError('Ordered finite view render and camera receipt clocks required')
        if context is not None:
            temporal_body_context(context, decision_body, render_start_lower=lower,
                                  camera_receipt_time=camera_receipt_time,
                                  decision_time=decision_time)
    return PerViewTemporalBodyContext(visual_contexts, decision_body, render_start_lowers,
                                      float(camera_receipt_time), float(decision_time))


def missing_temporal_body_context(decision_body, *, decision_time, reason):
    if (not isinstance(decision_body, NavigationBodyObservation)
            or isinstance(decision_time, bool) or not isinstance(decision_time, (float, int))
            or not math.isfinite(decision_time) or decision_time < 0
            or decision_body.decision_monotonic_seconds != decision_time
            or decision_body.original_received_monotonic_seconds > decision_time
            or reason not in ('no_proven_prior_receipt', 'receipt_too_old', 'no_receipts',
                              'unavailable_historical_receipt')):
        raise ValueError('Valid current observation and explicit historical absence required')
    return MissingTemporalBodyContext(decision_body, float(decision_time), reason)


def temporal_body_context(visual_context, decision_body, *, camera_receipt_time,
                          render_start_lower, decision_time):
    if (not isinstance(visual_context, NavigationBodyObservation)
            or not isinstance(decision_body, NavigationBodyObservation)):
        raise ValueError('Two original navigation-body observations required')
    clocks = (camera_receipt_time, render_start_lower, decision_time,
              visual_context.original_received_monotonic_seconds,
              decision_body.original_received_monotonic_seconds)
    if (not all(isinstance(x, (float, int)) and not isinstance(x, bool)
                and math.isfinite(x) and x >= 0 for x in clocks)
            or not (visual_context.original_received_monotonic_seconds < render_start_lower
                    <= camera_receipt_time <= decision_time)
            or decision_body.original_received_monotonic_seconds > decision_time
            or visual_context.decision_monotonic_seconds != camera_receipt_time
            or decision_body.decision_monotonic_seconds != decision_time
            or visual_context.episode_id != decision_body.episode_id
            or visual_context.contract_sha256 != decision_body.contract_sha256):
        raise ValueError('Ordered original host clocks, episode and body contract required')
    return TemporalBodyContext(visual_context, decision_body, float(camera_receipt_time),
                               float(render_start_lower), float(decision_time))


@dataclass(frozen=True)
class HostFreshnessLimits:
    """Decision-time host limits only; never an exposure or physical-time certificate."""

    maximum_body_receipt_age_seconds: float
    maximum_view_render_age_seconds: float

    def __post_init__(self):
        body, view = self.maximum_body_receipt_age_seconds, self.maximum_view_render_age_seconds
        if (isinstance(body, bool) or isinstance(view, bool)
                or not isinstance(body, (int, float)) or not isinstance(view, (int, float))
                or not math.isfinite(body) or not math.isfinite(view)
                or not 0 < body <= .05 or not 0 < view <= .15):
            raise ValueError('Explicit bounded host freshness limits required')


def require_decision_host_freshness(context, limits, *, held_body_receipt=None):
    """Reject stale selected/held receipts or view bounds at decision time.

    A missing held receipt is allowed only for caller-identified first priming;
    the caller must separately check output-time freshness and actuation authority.
    """
    if not isinstance(context, PerViewTemporalBodyContext) or type(limits) is not HostFreshnessLimits:
        raise ValueError('Original per-view context and explicit limits required')
    now = context.decision_time
    body_age = now - context.decision_body.original_received_monotonic_seconds
    if not math.isfinite(body_age) or not 0 <= body_age <= limits.maximum_body_receipt_age_seconds:
        raise ValueError('Current body receipt stale at decision')
    if held_body_receipt is not None:
        if (isinstance(held_body_receipt, bool) or not isinstance(held_body_receipt, (int, float))
                or not math.isfinite(held_body_receipt)
                or not 0 <= now-held_body_receipt <= limits.maximum_body_receipt_age_seconds):
            raise ValueError('Held body receipt stale at decision')
    ages = [now - lower for lower in context.render_start_lowers]
    if not all(math.isfinite(age) and 0 <= age <= limits.maximum_view_render_age_seconds for age in ages):
        raise ValueError('View render bound stale at decision')
