"""Time-weighted yaw for the synthetic stationary-then-rotation stimulus."""
import math

import numpy as np


def synthetic_interval_yaw(start, end, *, stationary_seconds, yaw_rate):
    """Return mean yaw over an interval, including any stationary overlap.

    Times are stimulus-relative seconds. This is a synthetic-data contract,
    not a physical sensor timestamp or a validated vehicle yaw measurement.
    """
    values = (start, end, stationary_seconds, yaw_rate)
    if (any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) for x in values)
            or start < 0 or stationary_seconds < 0 or end <= start):
        raise ValueError('Finite ordered nonnegative synthetic times required')
    moving = max(0., end - max(start, stationary_seconds))
    return float(yaw_rate * moving / (end - start))


def synthetic_body_at(receipt_time, *, stationary_seconds, yaw_rate,
                      forward_speed_m_s, wheel_radius_m):
    """Return synthetic body21 values at the observation's own receipt time.

    The planar stimulus translates and rotates at constant prescribed rates
    after the stationary prefix. Wheel rates are an engineering assumption;
    this does not simulate vehicle dynamics or measured tyre slip.
    """
    values = (receipt_time, stationary_seconds, yaw_rate, forward_speed_m_s, wheel_radius_m)
    if (any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) for x in values)
            or receipt_time < 0 or stationary_seconds < 0
            or forward_speed_m_s < 0 or wheel_radius_m <= 0):
        raise ValueError('Finite nonnegative synthetic timing and speed, positive radius required')
    body = np.zeros(21, dtype=np.float64)
    if receipt_time > stationary_seconds:
        body[4] = yaw_rate
        body[16] = forward_speed_m_s
        body[17:21] = forward_speed_m_s / wheel_radius_m
    return body
