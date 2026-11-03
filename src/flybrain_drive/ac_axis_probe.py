"""Finite stationary virtual-game axis experiment; receipts never imply application."""

import math
import time

STAGES = (
    ('neutral_baseline', (0., 0., 1.)),
    ('steer_positive', (.1, 0., 1.)),
    ('steer_negative', (-.1, 0., 1.)),
    ('neutral_after_steer', (0., 0., 1.)),
    ('brake_half', (0., 0., .5)),
    ('brake_released', (0., 0., 0.)),
    ('throttle_small_brake_held', (0., .1, 1.)),
    ('neutral_final', (0., 0., 1.)),
)


def run_stationary_probe(session, read_physics, guard, emit, *, samples_per_stage=8,
                         period_seconds=.05, max_speed_kmh=2., sleep=time.sleep,
                         already_open=False):
    """No profile writes, button presses, shifting, policy inference or automatic driving.

    The caller must first check the exact installed profile/device/game identity.
    guard rechecks producer/profile before every request. Raw steering units and
    response delay are measured later rather than assumed here.
    """
    if (type(samples_per_stage) is not int or not 2 <= samples_per_stage <= 16
            or not math.isfinite(period_seconds) or not .02 <= period_seconds <= .1
            or not math.isfinite(max_speed_kmh) or not 0 < max_speed_kmh <= 2):
        raise ValueError('Explicit finite stationary probe budgets required')
    sent = 0
    try:
        guard()
        receipt = session.submit(0., 0., 1.) if already_open else session.open()
        sent += 1
        emit(dict(kind='open', request=receipt))
        for stage, command in STAGES:
            for index in range(samples_per_stage):
                guard()
                sample_before = read_physics()
                if abs(sample_before.fields['speedKmh']) > max_speed_kmh:
                    raise RuntimeError('Stationary speed budget exceeded before request')
                receipt = session.submit(*command)
                sent += 1
                sample_after = read_physics()
                emit(dict(kind='axis_sample', stage=stage, index=index, request=receipt,
                          physics_before=sample_before, physics_after=sample_after))
                if abs(sample_after.fields['speedKmh']) > max_speed_kmh:
                    raise RuntimeError('Stationary speed budget exceeded after request')
                if stage == 'neutral_baseline' and (
                        sample_after.fields['gas'] > .02 or sample_after.fields['brake'] < .9):
                    raise RuntimeError('Requested neutral state not observed in native feedback')
                sleep(period_seconds)
        return dict(status='completed', experiment_requests=sent,
                    response_verified=False, autonomous_driving_verified=False)
    finally:
        # Records a request, not a promise that the game applied a stopping command.
        receipt = session.close()
        if receipt is not None:
            emit(dict(kind='close', request=receipt))
