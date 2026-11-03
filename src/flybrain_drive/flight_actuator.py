"""Explicit engineering intent-to-axis codec; no SDK or actuator side effects."""

import numpy as np


def canonical_controls(intent):
    """Left-turn intent positive, AC canonical steering negative; signed drive -> pedals."""
    values = np.asarray(intent, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(abs(values) > 1):
        raise ValueError("Two finite bounded flight intentions required")
    left, forward = values
    return np.array([-left, max(forward, 0.), max(-forward, 0.)])


def stationary_steering_request(intent, *, steering_scale=.6, maximum_steering=.1):
    """Steering-only component probe: retain raw output; explicitly override both pedals."""
    if (not np.isfinite([steering_scale, maximum_steering]).all()
            or not 0 < steering_scale <= .6 or not 0 < maximum_steering <= .1):
        raise ValueError("Declared stationary steering scale/cap required")
    canonical = canonical_controls(intent)
    scaled = float(canonical[0] * steering_scale)
    steering = float(np.clip(scaled, -maximum_steering, maximum_steering))
    return dict(raw_intent=np.asarray(intent, dtype=float).tolist(),
        canonical_control=canonical.tolist(), steering_scale=steering_scale,
        unclipped_scaled_steering=scaled, steering_limited=steering != scaled,
        requested_control=dict(steering=steering, throttle=0., brake=1.),
        pedal_override="stationary full-brake component test, not learned longitudinal control")
