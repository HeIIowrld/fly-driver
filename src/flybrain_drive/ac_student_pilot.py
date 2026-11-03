"""Bounded native student trials; diagnostic pose never crosses the inference callback."""

import time
from dataclasses import dataclass

import numpy as np

from .ac_route_progress import RouteProgress

STUDENT_TRIAL_PROFILES = dict(
    first=dict(decisions=25, speed=10., displacement=1., gap=None, travel=None),
    tracking_validation=dict(decisions=150, speed=25., displacement=20., gap=3., travel=15.),
    tracking_30kmh=dict(decisions=300, speed=40., displacement=40., gap=3., travel=30.),
    tracking_60m=dict(decisions=600, speed=40., displacement=80., gap=3., travel=60.),
    tracking_120m=dict(decisions=700, speed=40., displacement=160., gap=3., travel=120.),
    tracking_240m=dict(decisions=1200, speed=40., displacement=320., gap=3., travel=240.),
    course_240m=dict(decisions=1200, speed=40., displacement=320., gap=None, travel=240.,
                     clearance=1., tyres=0, valid_lap=True),
    course_480m=dict(decisions=2400, speed=40., displacement=640., gap=None, travel=480.,
                     clearance=1., tyres=0, valid_lap=True),
    game_valid_480m=dict(decisions=2400, speed=40., displacement=640., gap=None, travel=480.,
                         tyres=2, valid_lap=True),
    game_valid_route_480m=dict(decisions=2400, speed=40., displacement=640., gap=None,
                             travel=480., metric='sdk_route', tyres=2, valid_lap=True),
    receipt15_route60=dict(decisions=800, speed=20., displacement=90., gap=None,
        travel=60., metric='sdk_route', tyres=0, valid_lap=True, clearance=1.,
        task='applied_receipt', task_speed=15., camera_bytes=1024**3),
    receipt15_route240=dict(decisions=2600, speed=20., displacement=320., gap=None,
        travel=240., metric='sdk_route', tyres=0, valid_lap=True, clearance=1.,
        task='applied_receipt', task_speed=15., camera_bytes=2*1024**3),
    receipt15_route480=dict(decisions=2900, speed=20., displacement=640., gap=None,
        travel=480., metric='sdk_route', tyres=0, valid_lap=True, clearance=1.,
        task='applied_receipt', task_speed=15., camera_bytes=2*1024**3))

@dataclass(frozen=True)
class StudentTrialReading:
    native: object
    diagnostic: dict


def has_fixed_20kmh_task(contract):
    return has_fixed_speed_task(contract, 20.)


def has_fixed_speed_task(contract, speed_kmh):
    if not isinstance(contract, dict):
        return False
    return (contract.get('kind') == 'geometric_recovery_fixed_task'
            and contract.get('maximum_speed_kmh') == speed_kmh) or (
        contract.get('kind') == 'executed_sac_speed_governed'
        and contract.get('collection_speed_kmh') == speed_kmh)


def has_applied_receipt_task(contract, speed_kmh, steering_scale):
    """Separate experimental task; never grants old geometric/speed qualifications."""
    return (isinstance(contract, dict)
        and contract.get('kind') == 'engine_ai_future_applied_control_receipt_v1'
        and contract.get('ai_speed_hint_kmh') == speed_kmh
        and contract.get('ai_steer_multiplier') == .9
        and contract.get('actor_steering_device_scale') == steering_scale == .6
        and contract.get('target_receipt_horizon_seconds') == .04
        and contract.get('maximum_target_lateness_seconds') == .025
        and contract.get('clock') == 'host_monotonic_QPC_time.perf_counter'
        and contract.get('physical_command_time_verified') is False)


def check_trial_reading(reading, origin=None, *, max_speed_kmh=5., max_displacement_m=.5,
                        require_start=False, max_reference_gap_m=None,
                        min_lateral_clearance_m=None, max_tyres_out=2, require_valid_lap=False):
    if type(max_tyres_out) is not int or not 0 <= max_tyres_out <= 2:
        raise ValueError('Declared tyre envelope in 0..2 required')
    d = reading.diagnostic
    values = [d[k] for k in ('speed_kmh', 'gear', 'tyres_out', 'x', 'y')]
    if (not np.isfinite(values).all() or abs(d['speed_kmh']) > max_speed_kmh
            or d['gear'] < 2 or not 0 <= d['tyres_out'] <= max_tyres_out):
        raise ValueError(f'Student trial motion/gear/track envelope exceeded: {d}')
    if require_start and abs(d['speed_kmh']) > .5:
        raise ValueError('Student trial requires an already stationary car')
    if max_reference_gap_m is not None:
        gap, clearance = d.get('reference_gap_m'), d.get('minimum_lateral_clearance_m')
        if (gap is None or clearance is None or not np.isfinite([gap, clearance]).all()
                or abs(gap) > max_reference_gap_m or clearance < 1.):
            raise ValueError('Tracking validation reference-gap/lateral-clearance limit exceeded')
    if min_lateral_clearance_m is not None:
        clearance = d.get('minimum_lateral_clearance_m')
        if (not np.isfinite(min_lateral_clearance_m) or min_lateral_clearance_m < 1.
                or clearance is None or not np.isfinite(clearance)
                or clearance < min_lateral_clearance_m):
            raise ValueError('Course lateral-clearance envelope exceeded or unavailable')
    if require_valid_lap and d.get('lap_invalidated') != 0:
        raise ValueError('Course requires an explicitly valid game lap')
    if origin is not None:
        displacement = float(np.linalg.norm(np.array([d['x'], d['y']]) - origin))
        if displacement > max_displacement_m:
            raise ValueError(f'Student displacement {displacement:.6g} exceeds '
                             f'{max_displacement_m:.6g}m')


def run_student_trial(session, receive, infer, guard, emit, *, decisions=20,
                      steering_request_scale=.6, period=.04, max_speed_kmh=5.,
                      max_displacement_m=.5, max_inference_seconds=.08,
                      clock=time.monotonic, sleep=time.sleep, trial_profile='first',
                      schedule_clock=None, expert_handoff=None,
                      expert_handoff_minimum_route_m=0):
    """Inference receives only native body sample, never diagnostic position or SDK truth.

    The inference callback owns fresh images, pose-free transduction and persistent state.
    Requests are the unblended student outputs, with the declared steering axis conversion.
    Stop/identity guards and measured feedback precede each subsequent decision.
    """
    profiles = STUDENT_TRIAL_PROFILES
    if trial_profile not in profiles:
        raise ValueError('Explicit bounded student trial profile required')
    limits = profiles[trial_profile]
    if (type(expert_handoff_minimum_route_m) is not int
            or expert_handoff_minimum_route_m not in (0, 200, 320)
            or expert_handoff_minimum_route_m and expert_handoff is None):
        raise ValueError('Explicit launch,200m or320m moving-handoff context required')
    expected_handoff_profile = 'receipt15_route480' if expert_handoff_minimum_route_m else (
        'receipt15_route240')
    if expert_handoff is not None and (
            not callable(expert_handoff) or trial_profile != expected_handoff_profile
            or expert_handoff_minimum_route_m and decisions != 2000):
        raise ValueError('Explicit moving handoff is limited to its separate receipt-data trial')
    metric = limits.get('metric', 'displacement')
    if (type(decisions) is not int or not 1 <= decisions <= limits['decisions']
            or not np.isfinite([period, steering_request_scale, max_speed_kmh,
                               max_displacement_m, max_inference_seconds]).all()
            or period != .04 or not 0 < steering_request_scale <= 1
            or not 0 < max_speed_kmh <= limits['speed']
            or not 0 < max_displacement_m <= limits['displacement']
            or not 0 < max_inference_seconds <= .1):
        raise ValueError('Explicit finite first-student-trial limits required')
    def check(reading, origin=None, *, require_start=False, stationary=False):
        check_trial_reading(reading, origin,
            max_speed_kmh=.5 if stationary else max_speed_kmh,
            max_displacement_m=max_displacement_m, require_start=require_start,
            max_reference_gap_m=limits['gap'],
            min_lateral_clearance_m=limits.get('clearance'),
            max_tyres_out=limits.get('tyres', 2), require_valid_lap=limits.get('valid_lap', False))

    completed, expired = 0, None
    timer = schedule_clock or clock
    try:
        guard()
        current = receive()
        check(current, require_start=True)
        origin = np.array([current.diagnostic['x'], current.diagnostic['y']])
        route = RouteProgress(current.diagnostic, maximum_speed_kmh=max_speed_kmh) \
            if metric == 'sdk_route' else None
        if route is not None:
            emit(dict(kind='route_origin', diagnostic=current.diagnostic,
                      metric=metric, target_m=limits['travel']))
        emit(dict(kind='open', request=session.open()))
        sleep(.06)
        current = receive()
        check(current, origin, stationary=True)
        if (current.native.fields['gas'] > .02 or current.native.fields['brake'] < .98
                or abs(current.diagnostic['sdk_steer_raw']) > .5):
            raise ValueError('Actual game did not acknowledge neutral hold')
        cadence_origin = timer()
        deadline = cadence_origin + (decisions + 1) * period * 3
        for index in range(decisions):
            started = timer()
            guard()
            if started >= deadline:
                raise TimeoutError('Finite student trial wall-clock budget exceeded')
            check(current, origin)
            inferred = infer(current.native)
            value = np.asarray(inferred['prediction'], dtype=float)
            if (value.shape != (3,) or not np.isfinite(value).all() or abs(value[0]) > 1
                    or np.any((value[1:] < 0) | (value[1:] > 1))):
                raise ValueError('Student prediction outside normalized AC action contract')
            controls = value.copy()
            controls[0] *= steering_request_scale
            guard()
            now = clock()
            elapsed = timer()-started
            age = now - inferred['oldest_input_monotonic_seconds']
            if (not np.isfinite(age) or not 0 <= age <= .15
                    or elapsed > max_inference_seconds):
                expired = dict(kind='expired_inference', index=index,
                    input_age_seconds=float(age) if np.isfinite(age) else None,
                    inference_cycle_seconds=elapsed, maximum_input_age_seconds=.15,
                    maximum_inference_seconds=max_inference_seconds,
                    actor_input=inferred['metadata'], action_request_sent=False)
                raise TimeoutError('Student input/inference expired before action: '
                                   f'age={age:.6g}s, inference={elapsed:.6g}s')
            receipt = session.submit(*map(float, controls))
            emit(dict(kind='student_request', index=index, prediction=value.tolist(),
                request=receipt, actor_input=inferred['metadata'],
                inference_cycle_seconds=elapsed, input_age_before_request=age,
                steering_request_scale=steering_request_scale,
                teacher_action_used=False, diagnostic_pose_used_by_actor=False))
            # Feedback reads consume time too. Anchor all decisions to one episode
            # clock so each receive latency does not accumulate into the cadence.
            sleep(max(0., cadence_origin + (index + 1) * period - timer()))
            guard()
            current = receive()
            emit(dict(kind='feedback', index=index, diagnostic=current.diagnostic))
            check(current, origin)
            completed += 1
            travel = route.observe(current.diagnostic) if route is not None else float(
                np.linalg.norm(np.array([current.diagnostic['x'], current.diagnostic['y']])-origin))
            handoff_warning = expert_handoff is not None and (
                travel >= expert_handoff_minimum_route_m) and (
                current.diagnostic['speed_kmh'] >= 17.
                or current.diagnostic['speed_kmh'] >= 16.
                and current.diagnostic.get('minimum_lateral_clearance_m', float('inf')) <= 1.5)
            if handoff_warning:
                receipt = expert_handoff(session, current)
                if not isinstance(receipt, dict) or receipt.get('ai_active_observed') is not True:
                    raise ValueError('Explicit observed AI ownership required for handoff')
                emit(dict(kind='expert_handoff', index=index, receipt=receipt))
                return dict(status='expert_handoff', decisions=completed,
                    termination='explicit_expert_data_handoff', full_lap_verified=False,
                    student_closed_loop_trial=False, data_collection_with_expert=True)
            if limits['travel'] is not None and travel >= limits['travel']:
                emit(dict(kind='planned_stop', travelled_m=travel, target_m=limits['travel'],
                          metric=metric))
                return dict(status='segment_completed', decisions=completed,
                    termination='planned_travel_reached', full_lap_verified=False,
                    student_closed_loop_trial=True, trial_profile=trial_profile,
                    planned_travel_target_m=limits['travel'], planned_travel_reached=True,
                    measured_travel_m=travel, planned_travel_metric=metric)
        return dict(status='segment_completed', decisions=completed,
                    termination='decision_budget_exhausted', full_lap_verified=False,
                    student_closed_loop_trial=True, trial_profile=trial_profile,
                    planned_travel_target_m=limits['travel'],
                    planned_travel_reached=False if limits['travel'] is not None else None,
                    measured_travel_m=travel, planned_travel_metric=metric)
    finally:
        if session.owned:
            emit(dict(kind='close', request=session.close(), game_applied_verified=False))
        if expired is not None:
            # Diagnostic I/O must follow the neutral/brake request and driver release.
            emit(expired)
