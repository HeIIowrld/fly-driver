"""Declared low-speed engineering harness around flight steering, not a learned driver."""

import numpy as np

from .flight_actuator import canonical_controls


def moving_component_request(intent, *, phase, speed_kmh):
    """Separate model steering from the fixed low-speed pedal experiment."""
    if (not np.isfinite(speed_kmh) or not 0 <= speed_kmh < 10
            or phase not in ('stationary_prefix', 'launch', 'left_task_command',
                             'right_task_command', 'return_to_visual', 'braking')):
        raise ValueError('Fresh speed below10kmh and declared component phase required')
    canonical = canonical_controls(intent)
    steering = float(np.clip(canonical[0]*.6, -.1, .1))
    if phase in ('stationary_prefix', 'braking'):
        steering, throttle, brake = 0., 0., 1.
    else:
        throttle = float(np.clip(.03+.025*(6.-speed_kmh), 0., .12))
        brake = float(np.clip(.03*(speed_kmh-6.4), 0., .2))
        if brake > 0:
            throttle = 0.
    return dict(raw_intent=np.asarray(intent, dtype=float).tolist(),
                canonical_control=canonical.tolist(), steering_scale=.6,
                requested_control=dict(steering=steering, throttle=throttle, brake=brake),
                pedal_override='engineering6kmh speed hold; not learned longitudinal control',
                steering_override=phase in ('stationary_prefix', 'braking'))
