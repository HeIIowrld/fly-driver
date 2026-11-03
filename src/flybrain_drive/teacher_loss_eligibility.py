"""Pure diagnostic loss eligibility; never grants fitting or changes recorded history.

Callers must separately qualify API start/finish bounds, clock registration and
calibration provenance. Structural inputs here do not prove those qualifications.
The projected native clock is a HOST receipt bound, not a physical sample time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .ac_teacher_physics import TeacherPhysicsSample, decode_teacher_physics

VIEWS = frozenset({'front', 'left', 'right', 'rear'})


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class SourceInterval:
    generation: str
    lower: float
    upper: float

    def __post_init__(self):
        if (not isinstance(self.generation, str) or not self.generation
                or not _finite(self.lower) or not _finite(self.upper)
                or self.lower > self.upper):
            raise ValueError('Explicit generation and finite ordered source bounds required')


@dataclass(frozen=True)
class TeacherActivity:
    enable_call_bounds: SourceInterval | None
    disable_call_bounds: SourceInterval | None


@dataclass(frozen=True)
class TaskPhase:
    phase_id: str
    interval: SourceInterval

    def __post_init__(self):
        if (not isinstance(self.phase_id, str) or not self.phase_id
                or not isinstance(self.interval, SourceInterval)):
            raise ValueError('Unique declared phase identity and source interval required')


@dataclass(frozen=True)
class NativeControlCalibration:
    reference: str
    generation: str
    steering_native_per_canonical: float
    max_abs_canonical_steering: float
    pedal_min_positive: float

    def __post_init__(self):
        if (not isinstance(self.reference, str) or not self.reference
                or not isinstance(self.generation, str) or not self.generation
                or not _finite(self.steering_native_per_canonical)
                or self.steering_native_per_canonical <= 0
                or not _finite(self.max_abs_canonical_steering)
                or not 0 < self.max_abs_canonical_steering <= 1
                or not _finite(self.pedal_min_positive)
                or not 0 < self.pedal_min_positive <= 1
                or not math.isfinite(self.steering_native_per_canonical
                                     * self.max_abs_canonical_steering)):
            raise ValueError('Explicit finite externally reviewed calibration domain required')


@dataclass(frozen=True)
class ProjectedHostReceipt:
    original_host_receipt_seconds: float
    source_interval: SourceInterval
    clock_record_sha256: str

    def __post_init__(self):
        if (not _finite(self.original_host_receipt_seconds)
                or self.original_host_receipt_seconds < 0
                or not isinstance(self.source_interval, SourceInterval)
                or not isinstance(self.clock_record_sha256, str)
                or re.fullmatch('[0-9a-f]{64}', self.clock_record_sha256) is None):
            raise ValueError('Original host receipt, source interval and clock reference required')


def project_host_receipt_interval(host_receipt_seconds, clock_record, *, generation):
    """Recompute saved ping bounds, without a new clock read or extrapolation.

    Source generation and source-rate qualification remain the caller's duty.
    A record from a host-QPC fixture does not qualify an actual CSP clock.
    """
    if (not _finite(host_receipt_seconds) or host_receipt_seconds < 0
            or not isinstance(clock_record, Mapping)
            or clock_record.get('generation') != generation):
        raise ValueError('Original host receipt and matching saved clock generation required')
    quantum = clock_record.get('host_clock_quantization_seconds', 0.)
    pings = clock_record.get('exchanges')
    if (not _finite(quantum) or not 0 <= quantum <= .001
            or not isinstance(pings, list) or len(pings) < 2):
        raise ValueError('At least two original finite clock exchanges required')
    lower, upper, sent, received = [], [], [], []
    for row in pings:
        if not isinstance(row, Mapping):
            raise ValueError('Original clock exchange mapping required')
        a, b, source = (row.get(k) for k in ('sent', 'received', 'producer_clock'))
        if (not all(_finite(x) for x in (a, b, source)) or not 0 <= a <= b or source < 0):
            raise ValueError('Finite original causal clock exchange required')
        lower.append(a-source-quantum)
        upper.append(b-source+quantum)
        sent.append(a)
        received.append(b)
    lo, hi = max(lower), min(upper)
    if (lo > hi or hi-lo > .1
            or not all(_finite(clock_record.get(k)) for k in (
                'offset_lower_seconds', 'offset_upper_seconds'))
            or abs(lo-clock_record['offset_lower_seconds']) > 1e-10
            or abs(hi-clock_record['offset_upper_seconds']) > 1e-10
            or not min(sent) <= host_receipt_seconds <= max(received)):
        raise ValueError('Saved clock intersection differs or host receipt is outside ping span')
    fingerprint = hashlib.sha256(json.dumps(dict(clock_record), sort_keys=True,
                                           allow_nan=False).encode()).hexdigest()
    return ProjectedHostReceipt(host_receipt_seconds,
        SourceInterval(generation, host_receipt_seconds-hi-quantum,
                       host_receipt_seconds-lo+quantum), fingerprint)


@dataclass(frozen=True)
class LossEligibility:
    steering: bool
    throttle: bool
    brake: bool
    steering_reasons: tuple[str, ...]
    throttle_reasons: tuple[str, ...]
    brake_reasons: tuple[str, ...]
    temporal_reasons: tuple[str, ...]
    fit_allowed: bool = field(default=False, init=False)
    yaw_reference_policy: str = field(default='separate_unresolved', init=False)
    scope: str = field(default='diagnostic_native_control_loss_only', init=False)
    exact_native_physical_clock_verified: bool = field(default=False, init=False)
    exclusive_teacher_control_verified: bool = field(default=False, init=False)

    @property
    def channel_mask(self):
        return self.steering, self.throttle, self.brake


def _inside(interval, lower, upper):
    # Touching a transition boundary is deliberately uncertain.
    return lower < interval.lower and interval.upper < upper


def diagnostic_control_loss_eligibility(
    *,
    sample: TeacherPhysicsSample,
    camera_intervals: Mapping[str, SourceInterval] | None,
    native_host_receipt: ProjectedHostReceipt | None,
    activity: TeacherActivity | None,
    task_phases: Sequence[TaskPhase],
    expected_phase_id: str | None,
    calibration: NativeControlCalibration | None,
) -> LossEligibility:
    """Return per-channel necessary loss evidence; never create targets or a dataset.

    Missing action bounds, including historical post-call points substituted for
    start/finish intervals, are unknown. No raw observation or state is removed.
    Raw coupled AI=true is necessary, but does not establish exclusive control.
    """
    if not isinstance(sample, TeacherPhysicsSample):
        raise ValueError('Actual coupled native body/flag bytes required')
    if (type(sample.physics.packet_id) is not int
            or not _finite(sample.physics.received_monotonic_seconds)
            or sample.physics.received_monotonic_seconds < 0
            or type(sample.physics.prefix_bytes) is not bytes):
        raise ValueError('Original immutable actor bytes and finite native receipt required')
    verified = decode_teacher_physics(sample.authority_prefix_bytes,
                                     sample.physics.received_monotonic_seconds)
    if type(sample.builtin_ai_controlled) is not bool or verified != sample:
        raise ValueError('Native flag/body/receipt does not match the same420-byte copy')
    if calibration is not None and not isinstance(calibration, NativeControlCalibration):
        raise ValueError('Explicit calibration domain or None required')
    temporal = []
    if not sample.builtin_ai_controlled:
        temporal.append('native_AI_flag_false')
    if camera_intervals is not None and not isinstance(camera_intervals, Mapping):
        raise ValueError('Four view intervals or explicit missing evidence required')
    cameras = {} if camera_intervals is None else dict(camera_intervals)
    if set(cameras) - VIEWS or any(not isinstance(v, SourceInterval) for v in cameras.values()):
        raise ValueError('Only declared four-view source intervals are permitted')
    if set(cameras) != VIEWS:
        temporal.append('camera_intervals_incomplete')
    if native_host_receipt is not None and not isinstance(native_host_receipt,
                                                         ProjectedHostReceipt):
        raise ValueError('Explicit projected HOST receipt or None required')
    intervals = list(cameras.values())
    if native_host_receipt is None:
        temporal.append('native_HOST_receipt_bounds_unavailable')
    else:
        intervals.append(native_host_receipt.source_interval)
        if native_host_receipt.original_host_receipt_seconds != (
                sample.physics.received_monotonic_seconds):
            temporal.append('native_receipt_attachment_mismatch')
    generation, active = None, None
    if activity is None:
        temporal.append('activity_bounds_unavailable')
    elif not isinstance(activity, TeacherActivity):
        raise ValueError('Explicit API action intervals or None required')
    else:
        enable, disable = activity.enable_call_bounds, activity.disable_call_bounds
        if enable is None or disable is None:
            temporal.append('API_call_bounds_incomplete')
        elif not isinstance(enable, SourceInterval) or not isinstance(disable, SourceInterval):
            raise ValueError('Source bounds for API calls or explicit None required')
        elif (enable.lower == enable.upper or disable.lower == disable.upper
                or enable.upper >= disable.lower):
            temporal.append('API_call_bounds_incomplete')
        elif enable.generation != disable.generation:
            temporal.append('generation_mismatch')
        else:
            generation, active = enable.generation, (enable.upper, disable.lower)
    if active is not None:
        if any(v.generation != generation for v in intervals):
            temporal.append('generation_mismatch')
        if any(not _inside(v, *active) for v in intervals):
            temporal.append('whole_camera_or_HOST_receipt_outside_teacher_active_window')
    if (not isinstance(task_phases, Sequence)
            or any(not isinstance(v, TaskPhase) for v in task_phases)):
        raise ValueError('Explicit ordered phase intervals required')
    phases = list(task_phases)
    ambiguous = (len({v.phase_id for v in phases}) != len(phases)
                 or any(a.interval.upper > b.interval.lower
                        for a, b in zip(phases, phases[1:], strict=False)))
    if ambiguous:
        temporal.append('phase_declaration_ambiguous')
    if generation is not None and any(v.interval.generation != generation for v in phases):
        temporal.append('phase_generation_unqualified')
    phase = next((v for v in phases if v.phase_id == expected_phase_id), None)
    if phase is None:
        temporal.append('expected_phase_unavailable')
    elif (generation is None or phase.interval.generation != generation
          or any(v.generation != phase.interval.generation for v in intervals)):
        temporal.append('phase_generation_unqualified')
    elif any(not _inside(v, phase.interval.lower, phase.interval.upper) for v in intervals):
        temporal.append('whole_camera_or_HOST_receipt_phase_ambiguous')
    reasons = [list(dict.fromkeys(temporal)) for _ in range(3)]
    if calibration is None:
        for row in reasons:
            row.append('calibration_unavailable')
    elif generation != calibration.generation:
        for row in reasons:
            row.append('calibration_generation_mismatch')
    else:
        steer, gas, brake = (verified.physics.fields[k] for k in ('steerAngle', 'gas', 'brake'))
        if abs(steer) > 1 or abs(steer) > (calibration.steering_native_per_canonical
                                         * calibration.max_abs_canonical_steering):
            reasons[0].append('steering_outside_reviewed_native_calibration_domain')
        for index, value in [(1, gas), (2, brake)]:
            if value != 0 and not calibration.pedal_min_positive <= value <= 1:
                reasons[index].append('pedal_outside_reviewed_native_calibration_domain')
        if gas > 0 and brake > 0:
            reasons[1].append('pedal_coactivation_unrepresentable')
            reasons[2].append('pedal_coactivation_unrepresentable')
    return LossEligibility(*(not row for row in reasons),
        *(tuple(row) for row in reasons), tuple(dict.fromkeys(temporal)))
