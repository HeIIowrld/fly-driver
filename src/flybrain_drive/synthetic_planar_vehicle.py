"""Small synthetic vehicle for closed-loop experiments, not Assetto Corsa physics."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlanarVehicleState:
    x_m: float = 0.
    y_left_m: float = 0.
    yaw_left_rad: float = 0.
    speed_m_s: float = 0.


def advance_planar_vehicle(state, *, steer_rad, acceleration_m_s2, dt_s, wheelbase_m=2.5):
    """Integrate a slip-free bicycle with constant steering and acceleration.

    The returned pose belongs to the synthetic world only and must not enter an
    actor. The motion stops at zero speed; reverse, inertia, loads and tyres are
    not modelled. Steering is front-wheel angle, positive left, in radians.
    """
    if not isinstance(state, PlanarVehicleState):
        raise ValueError('Synthetic planar state required')
    nums = (state.x_m, state.y_left_m, state.yaw_left_rad, state.speed_m_s,
            steer_rad, acceleration_m_s2, dt_s, wheelbase_m)
    if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in nums)
            or state.speed_m_s < 0 or not 0 < dt_s <= .2 or wheelbase_m <= 0
            or abs(steer_rad) > .6 or not -12 <= acceleration_m_s2 <= 6):
        raise ValueError('Finite synthetic state and bounded SI controls required')
    moving_time = min(dt_s, -state.speed_m_s / acceleration_m_s2) if acceleration_m_s2 < 0 else dt_s
    distance = max(0., state.speed_m_s * moving_time + .5 * acceleration_m_s2 * moving_time**2)
    speed = max(0., state.speed_m_s + acceleration_m_s2 * dt_s)
    curvature = math.tan(steer_rad) / wheelbase_m
    yaw = state.yaw_left_rad + curvature * distance
    if abs(curvature) < 1e-10:
        dx = distance * math.cos(state.yaw_left_rad)
        dy = distance * math.sin(state.yaw_left_rad)
    else:
        dx = (math.sin(yaw) - math.sin(state.yaw_left_rad)) / curvature
        dy = (math.cos(state.yaw_left_rad) - math.cos(yaw)) / curvature
    return PlanarVehicleState(state.x_m + dx, state.y_left_m + dy, yaw, speed)
