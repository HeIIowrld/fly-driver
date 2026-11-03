"""Explicit synthetic crosswind perturbation around the slip-free bicycle model."""
import math
from dataclasses import replace
from .synthetic_planar_vehicle import PlanarVehicleState,advance_planar_vehicle

def advance_with_crosswind(state, *, steer_rad, acceleration_m_s2, dt_s,
                           wind_left_m_s, drift_gain):
    """Add fixed world-left drift; an engineering test, not vehicle aerodynamics."""
    if (type(wind_left_m_s) not in (int,float) or not math.isfinite(wind_left_m_s)
            or abs(wind_left_m_s)>20 or type(drift_gain) not in (int,float)
            or not math.isfinite(drift_gain) or not 0<=drift_gain<=.1):
        raise ValueError('Finite bounded synthetic wind and drift gain required')
    result=advance_planar_vehicle(state,steer_rad=steer_rad,acceleration_m_s2=acceleration_m_s2,dt_s=dt_s)
    return replace(result,y_left_m=result.y_left_m+wind_left_m_s*drift_gain*dt_s)
