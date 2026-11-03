import pytest
from flybrain_drive.synthetic_planar_vehicle import PlanarVehicleState,advance_planar_vehicle
from flybrain_drive.synthetic_crosswind_vehicle import advance_with_crosswind

def test_zero_wind_preserves_original_and_opposites_are_symmetric():
    state=PlanarVehicleState(x_m=3.,y_left_m=.1,yaw_left_rad=.1,speed_m_s=3.)
    args=dict(steer_rad=.02,acceleration_m_s2=0.,dt_s=.04)
    original=advance_planar_vehicle(state,**args)
    assert advance_with_crosswind(state,wind_left_m_s=0.,drift_gain=.02,**args)==original
    a=advance_with_crosswind(state,wind_left_m_s=5.,drift_gain=.02,**args)
    b=advance_with_crosswind(state,wind_left_m_s=-5.,drift_gain=.02,**args)
    assert a.y_left_m-original.y_left_m==pytest.approx(original.y_left_m-b.y_left_m)
    assert a.yaw_left_rad==b.yaw_left_rad and a.x_m==b.x_m

def test_unbounded_disturbance_rejected():
    with pytest.raises(ValueError):advance_with_crosswind(PlanarVehicleState(),steer_rad=0.,acceleration_m_s2=0.,dt_s=.04,wind_left_m_s=21.,drift_gain=.02)
