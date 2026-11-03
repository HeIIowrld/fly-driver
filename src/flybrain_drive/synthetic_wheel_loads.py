"""Hypothetical normalized load transfer for synthetic bicycle experiments only."""
import math
import numpy as np

def hypothetical_wheel_loads(*, speed_m_s, steer_rad, acceleration_m_s2,
                              wheelbase_m=2.5, transfer_gain=.2):
    values=(speed_m_s,steer_rad,acceleration_m_s2,wheelbase_m,transfer_gain)
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values):
        raise ValueError('Finite explicit vehicle values required')
    if speed_m_s<0 or wheelbase_m<=0 or not 0<=transfer_gain<=.5 or abs(steer_rad)>.7:
        raise ValueError('Explicit bounded synthetic parameters required')
    lateral=speed_m_s**2*math.tan(steer_rad)/wheelbase_m
    front=float(np.clip(.5-transfer_gain*acceleration_m_s2/9.81,.1,.9))
    left=float(np.clip(.5-transfer_gain*lateral/9.81,.1,.9))
    return np.array([front*left,front*(1-left),(1-front)*left,(1-front)*(1-left)],dtype=np.float64)
