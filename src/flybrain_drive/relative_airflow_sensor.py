"""Fail-closed engineering projection of simulator wind into body-relative airflow.

World vectors exist only at this boundary; the returned sensor has no pose or heading.
A source contract must be independently qualified before any axis can become valid.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirflowObservation:
    velocity_body_m_s: np.ndarray
    valid: np.ndarray
    source_verified: bool
    decision_time: float


def missing_airflow(decision):
    """Represent absent or unqualified airflow without inventing source receipts.

    The zero payload is only a masked storage value, never a claim of zero wind.
    A later decision must obtain a new observation for that exact decision time.
    """
    if (isinstance(decision, bool) or not isinstance(decision, (int, float))
            or not np.isfinite(decision) or decision < 0):
        raise ValueError('Finite nonnegative decision time required for missing airflow')
    output = np.zeros(3, dtype=np.float32)
    mask = np.zeros(3, dtype=bool)
    output.setflags(write=False)
    mask.setflags(write=False)
    return AirflowObservation(output, mask, False, float(decision))


def relative_airflow(*, wind_world_m_s, vehicle_world_m_s, body_axes_world,
                     wind_receipt, body_receipt, decision, source_verified,
                     maximum_age_seconds=.05, maximum_pair_skew_seconds=.01):
    """Project (air velocity minus car velocity) onto front/left/up unit axes.

    This is not a physical anemometer model. `source_verified` must attest units,
    vector sign/axes, generation and whether world wind applies at the vehicle.
    Receipt bounds cannot establish physics timestamps or atomic source snapshots.
    """
    if type(source_verified) is not bool:
        raise ValueError('Explicit independently verified source status required')
    if (not np.isfinite([wind_receipt, body_receipt, decision, maximum_age_seconds,
                         maximum_pair_skew_seconds]).all()
            or min(wind_receipt, body_receipt, decision) < 0
            or not 0 < maximum_age_seconds <= .05
            or not 0 <= maximum_pair_skew_seconds <= .01):
        raise ValueError('Finite causal clocks and bounded freshness required')
    wind, velocity, axes = (np.asarray(x, dtype=np.float64) for x in
                            (wind_world_m_s, vehicle_world_m_s, body_axes_world))
    if (wind.shape != (3,) or velocity.shape != (3,) or axes.shape != (3, 3)
            or not all(np.isfinite(x).all() for x in (wind, velocity, axes))
            or not np.allclose(axes@axes.T, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(axes), 1., atol=1e-5, rtol=0)):
        raise ValueError('Finite world velocities and right-handed front/left/up axes required')
    if max(wind_receipt, body_receipt) > decision:
        raise ValueError('Future source is not admissible')
    fresh = (decision-min(wind_receipt, body_receipt) <= maximum_age_seconds
             and abs(wind_receipt-body_receipt) <= maximum_pair_skew_seconds)
    ok = source_verified and fresh
    if ok:
        with np.errstate(over='ignore', invalid='ignore'):
            projected = axes @ (wind - velocity)
        if (not np.isfinite(projected).all()
                or np.any(np.abs(projected) > np.finfo(np.float32).max)):
            raise ValueError('Relative airflow must be finite and representable as float32')
        output = projected.astype(np.float32)
    else:
        output = np.zeros(3, np.float32)
    output.setflags(write=False)
    mask = np.full(3, ok, dtype=bool)
    mask.setflags(write=False)
    return AirflowObservation(output, mask, source_verified, float(decision))
