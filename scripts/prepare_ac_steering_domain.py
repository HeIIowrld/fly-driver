"""Offline steering-domain protocol helpers; no vehicle imports or live entry point.

These functions register a finite measurement schedule and expose inverse
ambiguity in hypothetical measured knots. They do not qualify a car or create
teacher labels. All steering output values remain native game-signal units.
"""

import math
from collections.abc import Sequence

CALIBRATION_MAGNITUDES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def signed_sweep(magnitudes: Sequence[float]):
    """One whole positive-out/return and negative-out/return history."""
    points = tuple(magnitudes)
    if (
        not points
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in points
        )
        or any(b <= a for a, b in zip(points, points[1:], strict=False))
    ):
        raise ValueError("Finite increasing normalized magnitudes in (0,1] required")
    rows = [dict(index=0, steering=0.0, branch="initial_neutral")]
    for sign, side in ((1, "positive"), (-1, "negative")):
        for branch, values in (("outward", points), ("return", (*reversed(points[:-1]), 0.0))):
            for value in values:
                rows.append(
                    dict(index=len(rows), steering=sign * value, branch=side + "_" + branch)
                )
    return rows


def validation_magnitudes():
    points = (0.0, *CALIBRATION_MAGNITUDES)
    return tuple((a + b) / 2 for a, b in zip(points, points[1:], strict=False))


def diagnostic_piecewise_preimage(requests, native_values, target):
    """Return all preimage intervals of an explicitly assumed knot interpolant.

    A flat saturation segment returns an interval, never a fabricated optimal
    command. Nonmonotonic or history-dependent curves need separate branches.
    No extrapolation is performed. This is an offline diagnostic, not a runtime
    decoder or permission to fit unqualified teacher controls.
    """
    xs, ys = tuple(requests), tuple(native_values)
    if (
        len(xs) != len(ys)
        or len(xs) < 2
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in (*xs, *ys, target))
        or any(not -1 <= x <= 1 for x in xs)
        or any(b <= a for a, b in zip(xs, xs[1:], strict=False))
        or any(b < a for a, b in zip(ys, ys[1:], strict=False))
    ):
        raise ValueError("Finite monotonic measured branch knots required")
    if not ys[0] <= target <= ys[-1]:
        return ()
    intervals = []
    for a, b, low, high in zip(xs, xs[1:], ys, ys[1:], strict=False):
        if not low <= target <= high:
            continue
        if low == high:
            interval = (a, b)
        else:
            point = (
                a
                if target == low
                else b
                if target == high
                else (a + (target - low) * (b - a) / (high - low))
            )
            interval = (point, point)
        if intervals and interval[0] <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], interval[1]))
        else:
            intervals.append(interval)
    return tuple(intervals)
