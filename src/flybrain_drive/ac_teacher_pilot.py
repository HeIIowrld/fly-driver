"""Finite privileged teacher driving, separate from the fly actor and from training."""

import math
import time

import numpy as np

from .ac_sac_teacher import legacy_request_to_normalized_controls, relative_request


def collection_speed_request(requested, speed_kmh, target_kmh):
    """Teacher-only bounded-speed curriculum, not the unmodified public SAC policy."""
    requested = np.asarray(requested, dtype=float)
    if (requested.shape != (3,) or not np.isfinite(requested).all()
            or np.any(np.abs(requested) > 1) or not np.isfinite([speed_kmh, target_kmh]).all()
            or not 10 <= target_kmh <= 90):
        raise ValueError('Explicit bounded collection speed and legacy requests required')
    controls = legacy_request_to_normalized_controls(requested)
    throttle_ceiling = float(np.clip((target_kmh-speed_kmh)/8., 0., 1.))
    brake_floor = float(np.clip((speed_kmh-target_kmh)/15., 0., .4))
    controls[1] = min(controls[1], throttle_ceiling)
    controls[2] = max(controls[2], brake_floor)
    if brake_floor > 0:
        controls[1] = 0.
    return np.array([controls[0], 2*controls[1]-1, 2*controls[2]-1])


def check_pilot_state(result, *, max_speed_kmh=60., max_gap_m=3., require_start=False):
    """On-track/gear gates, using privileged teacher state only."""
    state = result.state
    if (not np.isfinite([state['speed'], state['gap'], state['actualGear']]).all()
            or abs(state['speed']) * 3.6 > max_speed_kmh
            or abs(state['gap']) > max_gap_m or state['out_of_track']):
        raise ValueError('Teacher pilot speed/track envelope exceeded: '
            f'speed_kmh={state["speed"] * 3.6:.6g}/{max_speed_kmh:.6g}, '
            f'gap_m={state["gap"]:.6g}/{max_gap_m:.6g}, '
            f'out_of_track={state["out_of_track"]}')
    if state['actualGear'] < 2:
        raise ValueError('Forward gear required (SDK 1 is neutral); pilot never shifts')
    if require_start and abs(state['speed']) * 3.6 > .5:
        raise ValueError('Pilot must acquire controls from an already stationary car')


def validate_calibration(calibration, axis_config, producer_session_id, *, expected_gain=270.,
                         steering_request_scale=1.):
    """Require a completed stationary measurement from this exact profile and engine session."""
    if (calibration.get('status') != 'completed'
            or calibration.get('profile_sha256') != axis_config['profile_sha256']
            or calibration.get('producer_session_id') != producer_session_id
            or calibration.get('dll_sha256') != axis_config['dll_sha256']
            or calibration.get('device_id') != axis_config['device_id']
            or calibration.get('bindings') != axis_config['bindings']
            or calibration.get('samples') != 32):
        raise ValueError('Fresh same-session steering calibration for this profile required')
    gain = calibration.get('sdk_raw_per_normalized_request')
    medians = calibration.get('sdk_steering_stage_medians')
    amplitude = calibration.get('request_amplitude', .1)
    if (type(gain) not in (int, float) or not math.isfinite(gain)
            or not math.isfinite(expected_gain) or expected_gain <= 0
            or not math.isfinite(steering_request_scale) or not 0 < steering_request_scale <= 1
            or type(amplitude) not in (int, float) or not math.isfinite(amplitude)
            or not steering_request_scale <= amplitude <= 1
            or abs(gain * steering_request_scale / expected_gain - 1) > .02
            or not isinstance(medians, list) or len(medians) != 4
            or not np.isfinite(medians).all()
            or max(abs(medians[0]), abs(medians[3])) > .5
            or abs(medians[1] - amplitude * gain) > .5
            or abs(medians[2] + amplitude * gain) > .5):
        raise ValueError('Measured SDK steering scale/neutral differs from public teacher')


def run_teacher_segment(session, receive, teacher, codec, guard, emit, *, decisions=50,
                        period=.04, max_speed_kmh=60., max_gap_m=3.,
                        clock=time.monotonic, sleep=time.sleep, image_context=None,
                        steering_request_scale=1., reset_history=None,
                        collection_speed_kmh=None):
    """Callbacks must provide fresh assembled observations and enforce session/profile identity.

    The caller completes read-only calibration/car/track/clock checks first. Every request
    is followed by measured feedback. No retries, reset, focus, fabricated history or
    residual RL is performed. The final neutral request is logged even after failure.
    """
    if (type(decisions) is not int or not 1 <= decisions <= 500
            or not math.isfinite(period) or not .03 <= period <= .05
            or not 0 < max_speed_kmh <= 100 or not 0 < max_gap_m <= 5
            or not math.isfinite(steering_request_scale) or not 0 < steering_request_scale <= 1):
        raise ValueError('Bounded pilot budget required')
    if collection_speed_kmh is not None and (
            not math.isfinite(collection_speed_kmh)
            or not 10 <= collection_speed_kmh <= max_speed_kmh-10):
        raise ValueError('Collection speed must leave at least 10km/h below hard stop ceiling')
    completed = 0
    previous = np.array([0., -1., 1.])  # Same neutral state as VJoySession.open().
    try:
        # Read-only start check precedes even neutral acquisition.
        guard()
        current = receive()
        check_pilot_state(current, max_speed_kmh=max_speed_kmh,
                          max_gap_m=max_gap_m, require_start=True)
        if current.observation is None:
            raise ValueError('Three real teacher history rows required before acquisition')
        opening = session.open()
        deadline = clock() + period * (decisions + 4) * 1.5
        emit(dict(kind='open', request=opening))
        if reset_history is not None:
            reset_history('neutral_device_acquisition')
        # Refresh the real control history while the car is held neutral.
        for _ in range(4):
            sleep(period)
            guard()
            current = receive()
            check_pilot_state(current, max_speed_kmh=.5, max_gap_m=max_gap_m)
        actual = current.state
        if (abs(actual['steerAngle']) > .5 or actual['accStatus'] > .02
                or actual['brakeStatus'] < .98):
            raise ValueError('Game did not acknowledge neutral hold before teacher driving')
        for index in range(decisions):
            started = clock()
            if started >= deadline:
                raise TimeoutError('Finite teacher segment wall-clock budget exceeded')
            guard()
            check_pilot_state(current, max_speed_kmh=max_speed_kmh, max_gap_m=max_gap_m)
            if current.observation is None:
                raise ValueError('Real teacher history became unavailable')
            image = image_context(current) if image_context is not None else None
            delta = teacher.predict(current.observation[None])['deterministic_policy_action'][0]
            requested = relative_request(delta, previous, codec['legacy_control_max_delta'],
                [codec['legacy_control_bounds'][0]] * 3,
                [codec['legacy_control_bounds'][1]] * 3,
                product_dtype=codec['legacy_product_dtype'])
            unlimited = requested.copy()
            if collection_speed_kmh is not None:
                requested = collection_speed_request(requested, current.state['speed']*3.6,
                                                     collection_speed_kmh)
            controls = legacy_request_to_normalized_controls(requested)
            controls[0] *= steering_request_scale
            guard()
            receipt = session.submit(*map(float, controls))
            # Persist before waiting: failures still leave the last accepted request auditable.
            emit(dict(kind='teacher_request', index=index, observation=current.observation.tolist(),
                sdk_sequence=current.sdk_sequence, native_packet_id=current.native_packet_id,
                delta=delta.tolist(), legacy_absolute_request=requested.tolist(), request=receipt,
                ungoverned_legacy_request=unlimited.tolist(),
                collection_speed_kmh=collection_speed_kmh,
                speed_governor_changed_request=bool(np.any(requested != unlimited)),
                image_candidate=image, steering_request_scale=steering_request_scale))
            previous = requested
            remaining = period - (clock() - started)
            if remaining <= 0:
                raise TimeoutError('Teacher request exceeded decision budget')
            sleep(remaining)
            guard()
            current = receive()
            emit(dict(kind='feedback', index=index, sdk_sequence=current.sdk_sequence,
                native_packet_id=current.native_packet_id,
                sdk_steer_raw=current.state['steerAngle'],
                actual_throttle=current.state['accStatus'],
                actual_brake=current.state['brakeStatus'],
                speed_m_s=current.state['speed'], gap_m=current.state['gap'],
                actual_gear=current.state['actualGear'],
                out_of_track=current.state['out_of_track'],
                synchronized_exposure_verified=False))
            # Persist measured feedback even when it terminates the segment.
            check_pilot_state(current, max_speed_kmh=max_speed_kmh, max_gap_m=max_gap_m)
            completed += 1
        return dict(status='segment_completed', decisions=completed,
                    full_lap_verified=False, fly_actor_driving_verified=False)
    finally:
        if session.owned:
            closing = session.close()
            emit(dict(kind='close', request=closing, game_applied_verified=False))
