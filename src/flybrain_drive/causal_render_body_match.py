"""Conservative receipt-time matching; not physics or camera exposure synchronization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RenderBodyMatch:
    native_index: int | None
    maximum_host_age_seconds: float | None
    reason: str
    physical_synchronization_verified: bool = False


def select_prior_body_receipts_per_view(receipts, render_lower, camera_received,
                                        *, maximum_age_seconds=0.05):
    """Independently select a strictly prior receipt for each original view."""
    lower = np.asarray(render_lower, dtype=np.float64)
    if lower.shape != (4,) or not np.isfinite(camera_received):
        raise ValueError('Four view bounds and one finite camera receipt required')
    return tuple(select_prior_body_receipt(receipts, np.full(4, value),
        np.full(4, camera_received), maximum_age_seconds=maximum_age_seconds) for value in lower)


def select_prior_body_receipt(receipts, render_lower, render_upper, *, maximum_age_seconds=0.05):
    """Select latest receipt definitely before every render start, or explicitly reject it.

    All clocks must be host-monotonic seconds with already-reviewed offset bounds.
    Input receipt timestamps do not prove the physics source sample time. This helper
    is not an actor-admission or exposure-synchronization certificate.
    """
    times = np.asarray(receipts, dtype=np.float64)
    lower = np.asarray(render_lower, dtype=np.float64)
    upper = np.asarray(render_upper, dtype=np.float64)
    if (
        times.ndim != 1
        or lower.shape != (4,)
        or upper.shape != (4,)
        or not all(np.isfinite(x).all() for x in (times, lower, upper))
        or np.any(np.diff(times) <= 0)
        or np.any(lower > upper)
        or (times < 0).any()
        or (lower < 0).any()
        or not np.isfinite(maximum_age_seconds)
        or not 0 < maximum_age_seconds <= 0.05
    ):
        raise ValueError("Ordered host receipts, four finite time bounds and bounded age required")
    if not len(times):
        return RenderBodyMatch(None, None, "no_receipts")
    # Equal timestamps cannot establish that receipt preceded the render.
    index = int(np.searchsorted(times, lower.min(), side="left") - 1)
    if index < 0:
        return RenderBodyMatch(None, None, "no_proven_prior_receipt")
    maximum_age = float(upper.max() - times[index])
    if maximum_age > maximum_age_seconds:
        return RenderBodyMatch(None, maximum_age, "receipt_too_old")
    return RenderBodyMatch(index, maximum_age, "prior_receipt_within_host_age")
