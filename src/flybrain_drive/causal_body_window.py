"""Receipt-time-only body aggregation for diagnostics; no physical time claim."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriorWindow:
    value: float | None
    reason: str
    physical_synchronization_verified: bool = False


def prior_window_mean(receipts, values, *, end_lower, end_upper, duration, max_gap=.05):
    """Integrate the last held receipt over a prior host window, excluding end ties.

    `end_lower` and `end_upper` bound all render starts in host time. A missing prefix,
    stale last sample at the upper bound, or internal gap is invalid. This does not
    register a native axis or certify physics sampling or camera exposure timing.
    """
    times, data = np.asarray(receipts, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if (times.ndim != 1 or data.shape != times.shape or not np.isfinite(times).all()
            or not np.isfinite(data).all() or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.isfinite([end_lower, end_upper, duration, max_gap]).all()
            or not 0 < duration <= .2 or not 0 < max_gap <= .05 or end_lower < duration
            or end_upper < end_lower):
        raise ValueError("Ordered finite receipts, values and bounded prior window required")
    start = end_lower - duration
    first = int(np.searchsorted(times, start, side="right") - 1)
    stop = int(np.searchsorted(times, end_lower, side="left"))
    if first < 0 or stop <= first:
        return PriorWindow(None, "insufficient_prior_coverage")
    used_times = times[first:stop]
    if (start-used_times[0] > max_gap or end_upper-used_times[-1] > max_gap
            or np.any(np.diff(used_times) > max_gap)):
        return PriorWindow(None, "receipt_gap_too_large")
    change_times = used_times[1:]
    boundaries = np.r_[start, change_times, end_lower]
    return PriorWindow(float(np.dot(np.diff(boundaries), data[first:stop]) / duration), "ok")
