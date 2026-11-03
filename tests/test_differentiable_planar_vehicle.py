import math
import numpy as np
import pytest
import torch
from flybrain_drive.synthetic_planar_vehicle import PlanarVehicleState,advance_planar_vehicle
from flybrain_drive.differentiable_planar_vehicle import differentiable_bicycle_step

def test_matches_existing_synthetic_physics_including_stop_and_zero_curvature():
    for speed,steer,accel in [(3.,0.,0.),(6.,.2,-2.),(.02,-.3,-3.),(0.,0.,0.),(4.,1e-11,2.)]:
        state=PlanarVehicleState(2.,-1.,.4,speed);ref=advance_planar_vehicle(state,steer_rad=steer,acceleration_m_s2=accel,dt_s=.04)
        t=torch.tensor([[2.,-1.,.4,speed]],dtype=torch.float64);out=differentiable_bicycle_step(t,torch.tensor([steer],dtype=torch.float64),torch.tensor([accel],dtype=torch.float64),dt_s=.04)[0].numpy()
        assert np.allclose(out,[ref.x_m,ref.y_left_m,ref.yaw_left_rad,ref.speed_m_s],atol=1e-12,rtol=0)

def test_steering_gradient_matches_finite_difference():
    state=torch.tensor([[0.,0.,.1,4.]],dtype=torch.float64);s=torch.tensor([.08],dtype=torch.float64,requires_grad=True);a=torch.tensor([0.],dtype=torch.float64);y=differentiable_bicycle_step(state,s,a,dt_s=.04)[0,1];y.backward();eps=1e-6
    fn=lambda v:differentiable_bicycle_step(state,torch.tensor([v],dtype=torch.float64),a,dt_s=.04)[0,1].item()
    assert math.isclose(s.grad.item(),(fn(.08+eps)-fn(.08-eps))/(2*eps),rel_tol=1e-6)

def test_rejects_invalid_state():
    with pytest.raises(ValueError):differentiable_bicycle_step(torch.tensor([[0.,0.,0.,-1.]]),torch.tensor([0.]),torch.tensor([0.]),dt_s=.04)
