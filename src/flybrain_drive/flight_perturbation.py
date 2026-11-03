"""Predeclared steering disturbance and sampled physical-response metrics."""

import numpy as np


def validate_perturbation_plan(plan, index):
    expected = [('stationary_prefix', 25), ('launch', 100), ('baseline', 25),
                ('perturbation', 7), ('recovery', 50), ('braking', 50)]
    order = [(0, 'full', 1), (0, 'zero_yaw', 1), (0, 'zero_yaw', -1), (0, 'full', -1),
             (1, 'zero_yaw', 1), (1, 'full', 1), (1, 'full', -1), (1, 'zero_yaw', -1),
             (2, 'full', -1), (2, 'zero_yaw', -1), (2, 'zero_yaw', 1), (2, 'full', 1)]
    if (plan['schema'] not in ('flybrain.flight.physical_yaw_perturbation.v1',
                              'flybrain.flight.physical_yaw_perturbation.v2')
            or [(s['name'], s['samples']) for s in plan['stages']] != expected
            or any(s['task_turn_bias'] != 0 for s in plan['stages'])
            or [(t['block'], t['mode'], t['sign']) for t in plan['trials']] != order
            or type(index) is not int or not 0 <= index < 12
            or plan['perturbation_steering_amplitude'] != .05
            or plan['total_steering_limit'] != .1 or plan['task_speed_kmh'] != 6
            or plan['period_seconds'] != .04 or plan['maximum_speed_kmh'] != 10
            or plan['maximum_displacement_m'] != 20):
        raise ValueError('Exact predeclared12-trial physical comparison profile required')
    if plan['schema'].endswith('.v2') and (
            plan['timed_pulse_seconds'] != .28 or not plan['timed_actuator_verification']
            or not plan['scratch_warmup_verification']
            or plan['evaluation']['pair_pulse_width_tolerance_seconds'] != .01):
        raise ValueError('Certified280ms timer/warmup and10ms pair timing limit required')
    return plan['trials'][index]


def perturb_request(request, *, phase, sign):
    if phase not in ('stationary_prefix', 'launch', 'baseline', 'perturbation',
                     'recovery', 'braking') or type(sign) is not int or sign not in (-1, 1):
        raise ValueError('Declared phase and signed steering disturbance required')
    c = dict(request)
    if (set(c) != {'steering', 'throttle', 'brake'}
            or not np.isfinite(list(c.values())).all() or abs(c['steering']) > .1
            or not 0 <= c['throttle'] <= .12 or not 0 <= c['brake'] <= 1):
        raise ValueError('Bounded original component request required')
    pulse = sign*.05 if phase == 'perturbation' else 0.
    total = float(np.clip(c['steering']+pulse, -.1, .1))
    return dict(base_control=c, disturbance=pulse, unclipped_steering=c['steering']+pulse,
                clipped=total != c['steering']+pulse,
                requested_control=dict(c, steering=total))


def yaw_metrics(times, yaw, *, pulse_on, pulse_off, spec):
    """Use only original samples; recovery is an observed bound, not exact crossing time."""
    t, y = np.asarray(times, float), np.asarray(yaw, float)
    end = pulse_off+spec['recovery_seconds']
    if (t.ndim != 1 or y.shape != t.shape or len(t) < 10 or not np.isfinite(t).all()
            or not np.isfinite(y).all() or not np.all(np.diff(t) > 0)
            or not .20 <= pulse_off-pulse_on <= .36
            or t[0] > pulse_on-spec['baseline_seconds'] or t[-1] < end
            or np.diff(t).max() > spec['maximum_native_gap_seconds']):
        raise ValueError('Complete ordered original baseline/pulse/recovery samples required')
    baseline = y[(t >= pulse_on-spec['baseline_seconds']) & (t < pulse_on)]
    if len(baseline) < 5:
        raise ValueError('At least five actual baseline samples required')
    reference = float(np.mean(baseline))
    selected = (t >= pulse_on) & (t <= end)
    observed_t, error = t[selected], y[selected]-reference
    peak = float(np.abs(error).max())
    band = max(spec['settling_absolute_band_rad_s'], spec['settling_peak_fraction']*peak)
    recovery = None
    for index in np.flatnonzero(observed_t >= pulse_off):
        if (observed_t[-1]-observed_t[index] >= spec['minimum_settled_observation_seconds']
                and np.all(np.abs(error[index:]) <= band)):
            recovery = float(observed_t[index]-pulse_off)
            break
    delta_time = np.diff(observed_t)
    area = float(np.sum((np.abs(error[:-1])+np.abs(error[1:]))*.5*delta_time))
    signed = float(np.sum((error[:-1]+error[1:])*.5*delta_time))
    return dict(baseline_yaw_left_rad_s=reference, baseline_samples=len(baseline),
                iae_rad=area, peak_error_rad_s=peak,
                signed_yaw_integral_rad=signed,
                observed_window=[float(observed_t[0]), float(observed_t[-1])],
                observed_recovery_seconds_upper=recovery, settling_band_rad_s=band,
                recovery_observed=recovery is not None, physical_application_time_verified=False)
