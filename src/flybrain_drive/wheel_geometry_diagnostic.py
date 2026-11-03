"""Offline geometry analysis for non-actor CSP wheel diagnostics."""
import numpy as np

_EXPECTED = ("FL", "FR", "RL", "RR")


def analyze_wheel_geometry(diagnostic, *, minimum_axis_separation_m=0.1):
    """Classify four wheel centres in a car-relative basis, without admitting sensors.

    The basis and wheel positions are CSP diagnostics; a passing classification
    does not establish native shared-memory correspondence or physical timing.
    """
    if (not isinstance(diagnostic, dict)
            or diagnostic.get("schema") != "flybrain.csp.wheel-geometry.diagnostic.v1"
            or diagnostic.get("actor_admissible") is not False
            or diagnostic.get("source_verified") is not False
            or not np.isfinite(minimum_axis_separation_m) or minimum_axis_separation_m <= 0):
        raise ValueError("Explicit unverified diagnostic and positive threshold required")
    wheels = diagnostic.get("wheels")
    if not isinstance(wheels, list) or len(wheels) != 4 or [w.get("native_index") for w in wheels] != list(range(4)):
        raise ValueError("Four unique original-order wheels required")
    positions = np.asarray([w["position_world"] for w in wheels], dtype=float)
    contacts = np.asarray([w["contact_world"] for w in wheels], dtype=float)
    outside = np.asarray([w["outside_world"] for w in wheels], dtype=float)
    loads = np.asarray([w["load_candidate"] for w in wheels], dtype=float)
    forward = np.asarray(diagnostic["body_forward_candidate"], dtype=float)
    up = np.asarray(diagnostic["body_up_candidate"], dtype=float)
    if (positions.shape != (4, 3) or contacts.shape != (4, 3) or outside.shape != (4, 3)
            or loads.shape != (4,) or forward.shape != (3,) or up.shape != (3,)
            or not all(np.isfinite(x).all() for x in (positions, contacts, outside, loads, forward, up))
            or (loads < 0).any()):
        raise ValueError("Finite wheel geometry and nonnegative loads required")
    fn, un = np.linalg.norm(forward), np.linalg.norm(up)
    if fn < .9 or fn > 1.1 or un < .9 or un > 1.1:
        raise ValueError("Body axes require near-unit lengths")
    forward, up = forward / fn, up / un
    if abs(np.dot(forward, up)) > .05:
        raise ValueError("Body axes must be nearly orthogonal")
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    centered = positions - positions.mean(axis=0)
    longitudinal, lateral = centered @ forward, centered @ left
    if np.any(np.abs(longitudinal) < minimum_axis_separation_m) or np.any(np.abs(lateral) < minimum_axis_separation_m):
        raise ValueError("Wheel positions are not separated around the centre")
    labels = [("F" if x > 0 else "R") + ("L" if y > 0 else "R") for x, y in zip(longitudinal, lateral)]
    if len(set(labels)) != 4:
        raise ValueError("Wheel geometry does not yield four distinct quadrants")
    outside_norm = np.linalg.norm(outside, axis=1)
    if np.any(outside_norm < .9) or np.any(outside_norm > 1.1):
        raise ValueError("Wheel outside vectors require near-unit lengths")
    agreement = (outside / outside_norm[:, None]) @ left * np.sign(lateral)
    if np.any(agreement < .5):
        raise ValueError("Wheel outside vectors disagree with side geometry")
    return {"observed_labels": labels, "matches_fl_fr_rl_rr": labels == list(_EXPECTED),
            "longitudinal_m": longitudinal.tolist(), "lateral_m": lateral.tolist(),
            "outside_side_alignment": agreement.tolist(), "csp_geometry_verified": False,
            "native_order_registered": False, "actor_admissible": False}
