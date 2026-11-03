import pytest
import torch
from flybrain_drive.flight_state_speed_command import SymmetricFlightStateSpeedCommand


def test_neutral_symmetric_and_bounded_speed():
    net=SymmetricFlightStateSpeedCommand(6,max_delta_m_s=5.)
    muscles=torch.tensor([[.01,.03,.02,.04,.01,.02]])
    speed=torch.tensor([[3.]])
    goals=torch.tensor([[0.,.2]])
    assert net(muscles,speed,goals).tolist()==[[0.,pytest.approx(3.)]]
    with torch.no_grad():
        net.projection.weight.fill_(2.);net.projection.bias.fill_(2.)
    reflected=muscles.reshape(1,3,2).flip(-1).reshape(1,6)
    assert torch.equal(net(muscles,speed,goals),net(reflected,speed,goals))
    assert 0<=net(muscles,speed,goals)[0,1]<=.2*100/3.6+1e-6


def test_pair_scale_and_input_are_explicit():
    with pytest.raises(ValueError):SymmetricFlightStateSpeedCommand(5)
    with pytest.raises(ValueError):SymmetricFlightStateSpeedCommand(6,pair_scale=1e-8)
    with pytest.raises(ValueError):SymmetricFlightStateSpeedCommand(6,max_gain=100)
    net=SymmetricFlightStateSpeedCommand(6)
    with pytest.raises(ValueError):net(torch.full((1,6),float('nan')),torch.ones(1,1),torch.tensor([[0.,1.]]))
