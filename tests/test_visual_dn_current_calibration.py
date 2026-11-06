import pytest
import torch
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration,PowerMonotoneVehicleReadout

def make(rank):return VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]),rank=rank)

@pytest.mark.parametrize('rank',[1,4])
def test_neutral_exact_and_finite_gradient(rank):
 m=make(rank);x=torch.linspace(-2,2,640).reshape(2,320);c=torch.tensor([[0.,.001,.1,.2],[.05,.06,.08,.15]]);y=m(x,c);assert torch.equal(y,c);y.square().sum().backward();assert torch.isfinite(m.projection.weight.grad).all() and m.projection.weight.grad.abs().max()>0

@pytest.mark.parametrize('rank',[1,4])
def test_bounded_currents_for_large_weights(rank):
 m=make(rank)
 with torch.no_grad():m.projection.weight.fill_(100)
 y=m(torch.ones(2,320),torch.tensor([[0.,.001,.19,.2],[.1,.02,.08,.18]]));assert (y>=0).all() and (y<=.2).all()

def test_independent_ports_and_invalid_input():
 m=make(4)
 with torch.no_grad():m.projection.weight[1,0]=1
 x=torch.ones(1,320);c=torch.full((1,4),.02);y=m(x,c);assert y[0,1]>c[0,1] and torch.equal(y[:,[0,2,3]],c[:,[0,2,3]])
 with pytest.raises(ValueError):m(x,torch.full((1,4),float('nan')))

@pytest.mark.parametrize('power',[4,5])
def test_individual_power_sign(power):
 state=dict(training_reference=torch.zeros(6),training_scale=torch.ones(6),**{'projection.weight':torch.randn(3,6),'projection.bias':torch.zeros(3)});m=PowerMonotoneVehicleReadout(state);m.project();x=torch.full((2,6),.2,requires_grad=True);y=m(x);gas=torch.autograd.grad(y[:,1].sum(),x,retain_graph=True)[0];brake=torch.autograd.grad(y[:,2].sum(),x)[0];assert (gas[:,power]>0).all() and (brake[:,power]<0).all()
