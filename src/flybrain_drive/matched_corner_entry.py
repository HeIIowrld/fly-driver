"""Privileged matched-start check for a moving trial; never a policy input."""
import math


def native_motion_resolved(previous, current):
    return math.hypot(current[0]-previous[0], current[2]-previous[2]) >= .01


def corner_entry_measurement(plan, previous, current, *, now):
    """Use actual native position progression to reject mismatched entry poses."""
    center = tuple(plan["center_native_xyz_m"])
    tangent = tuple(plan["forward_xz"])
    speed_range = tuple(plan["speed_range_kmh"])
    along_max = plan["maximum_along_error_m"]
    lateral_max = plan["maximum_lateral_error_m"]
    direction_max = plan["maximum_direction_error_rad"]
    if (len(center) != 3 or len(tangent) != 2 or len(speed_range) != 2
            or not all(math.isfinite(v) for v in (*center,*tangent,*speed_range,
                                                along_max,lateral_max,direction_max,now))
            or not math.isclose(math.hypot(*tangent),1.,abs_tol=1e-6)
            or not 0 < speed_range[0] < speed_range[1]
            or not 0 < along_max <= 5 or not 0 < lateral_max <= 1
            or not 0 < direction_max <= .2):
        raise ValueError("Explicit narrow native geometry and speed match required")
    if any(row["generation"] != plan["generation"] or row["epoch"] != plan["epoch"]
           or row["track"] != plan["track"] or row["native_ai"] is not False
           or row["tyres_out"] != 0 for row in (previous,current)):
        raise ValueError("Native trial authority, track or tyre guard failed")
    then, stamp = previous["original_position_receipt"], current["original_position_receipt"]
    if (not all(math.isfinite(v) for v in (then,stamp))
            or not 0. < stamp-then <= .05 or not 0 <= now-stamp <= .05
            or current["packet_id"] <= previous["packet_id"]):
        raise ValueError("Fresh advancing original native positions required")
    point, older = (tuple(v["position_native_xyz_m"]) for v in (current,previous))
    if any(len(v)!=3 or not all(math.isfinite(x) for x in v) for v in (point,older)):
        raise ValueError("Finite native positions required")
    dx,dz = point[0]-center[0], point[2]-center[2]
    tx,tz = tangent
    along,side = dx*tx+dz*tz, dz*tx-dx*tz
    vx,vz = point[0]-older[0],point[2]-older[2]
    if not native_motion_resolved(older, point):
        raise ValueError("Native entry motion is not resolved")
    heading = math.atan2(vz*tx-vx*tz, vx*tx+vz*tz)
    speed = current["speed_kmh"]
    return dict(eligible=bool(math.isfinite(speed) and speed_range[0]<=speed<=speed_range[1]
        and abs(along)<=along_max and abs(side)<=lateral_max and abs(heading)<=direction_max),
        speed_kmh=speed, along_error_m=along, lateral_error_m=side, direction_error_rad=heading,
        original_position_receipt=stamp, actor_input=False)
