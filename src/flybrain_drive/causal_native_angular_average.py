"""Causal time average of native angular vectors, without axis registration.

This helper is a diagnostic sensor primitive. It does not mark any component
valid, infer physical acquisition times, or connect a vehicle controller.
"""
import numpy as np


def integrate_received_angular_velocity(receipts, vectors, *, start, end, decision, max_gap=.05):
    """Average a bracketed interval using only first receipts available by decision.

    No extrapolation or bridging long gaps is permitted. The returned values keep
    the original native axis order; units and physical synchronization remain the
    responsibility of the caller.
    """
    times = np.asarray(receipts, dtype=np.float64)
    values = np.asarray(vectors, dtype=np.float64)
    if (times.ndim != 1 or values.shape != (len(times), 3) or len(times) < 2
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or not np.all(np.diff(times) > 0)
            or not np.isfinite([start, end, decision, max_gap]).all()
            or not start < end <= decision or not 0 < max_gap <= .05):
        raise ValueError('Finite ordered first receipts and causal interval required')
    count = np.searchsorted(times, decision, side='right')
    times, values = times[:count], values[:count]
    left = np.searchsorted(times, start, side='right') - 1
    right = np.searchsorted(times, end, side='left')
    if left < 0 or right >= len(times):
        raise ValueError('Causally available packets must bracket the interval')
    brackets = times[left:right + 1]
    if np.any(np.diff(brackets) > max_gap):
        raise ValueError('Native packet gap exceeds the declared bound')
    inside = times[(times > start) & (times < end)]
    sample_times = np.r_[start, inside, end]
    sample_values = np.column_stack([
        np.interp(sample_times, times, values[:, axis]) for axis in range(3)
    ])
    return np.sum((sample_values[:-1] + sample_values[1:])
                  * np.diff(sample_times)[:, None] / 2, axis=0) / (end - start)
