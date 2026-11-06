import pytest
import torch
from flybrain_drive.paired_flight_vehicle_readout import PairedFlightVehicleReadout

def model():
 torch.manual_seed(7731);m=PairedFlightVehicleReadout(torch.ones(3)*.1,torch.zeros(4),torch.ones(4),torch.tensor(.4),torch.tensor(.1))
 with torch.no_grad():m.steering_weight.copy_(torch.tensor([.4,-.2,.3]));m.context_weight.uniform_(-.1,.1);m.long_bias.fill_(-.2);m.power_gain.copy_(torch.tensor([.5,.7]))
 return m

def test_reflection_flips_steer_and_preserves_both_pedals():
 m=model();x=torch.rand(100,6);flipped=x[:,[1,0,3,2,5,4]];a,b=m(x),m(flipped);torch.testing.assert_close(a[:,0],-b[:,0],rtol=0,atol=1e-6);torch.testing.assert_close(a[:,1:],b[:,1:],rtol=0,atol=1e-6)

def test_bilateral_equal_outputs_do_not_create_steering():
 m=model();x=torch.rand(100,3).repeat_interleave(2,dim=1);assert torch.count_nonzero(m(x)[:,0])==0

def test_common_power_has_no_steer_and_has_correct_pedal_direction():
 m=model();x=torch.rand(100,6)*.3+.2;y=x.clone();y[:,4:]+=.03;a,b=m(x),m(y);torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=2e-6);assert (b[:,1]>a[:,1]).all() and (b[:,2]<a[:,2]).all()

def test_projection_cannot_reintroduce_negative_power_gains():
 m=model()
 with torch.no_grad():m.power_gain.fill_(-3);m.steering_weight.fill_(100)
 m.project();assert (m.power_gain>=.02).all() and (m.steering_weight<=10).all()

def test_free_comparator_explicitly_allows_inversion():
 m=model();m.monotone=False
 with torch.no_grad():m.power_gain.fill_(-1.)
 m.project();assert (m.power_gain<0).all()

def test_common_power_gradient_is_nonnegative_gas_nonpositive_brake():
 m=model();x=torch.full((1,6),.4,requires_grad=True);output=m(x);g=torch.autograd.grad(output[0,1],x,retain_graph=True)[0];b=torch.autograd.grad(output[0,2],x)[0];assert g[0,4:].sum()>0 and b[0,4:].sum()<0

@pytest.mark.parametrize('fault',['shape','nan','negative','above_one','dtype'])
def test_invalid_muscle_observation_rejected(fault):
 m=model();x=torch.full((1,6),.4)
 if fault=='shape':x=torch.zeros(1,7)
 elif fault=='nan':x[0,0]=float('nan')
 elif fault=='negative':x[0,0]=-.1
 elif fault=='above_one':x[0,0]=1.1
 elif fault=='dtype':x=x.double()
 with pytest.raises(ValueError):m(x)

@pytest.mark.parametrize('side',[4,5])
def test_either_power_proxy_has_correct_pedal_direction(side):
 m=model();x=torch.rand(100,6)*.3+.2;y=x.clone();y[:,side]+=.03;a,b=m(x),m(y);assert (b[:,1]>a[:,1]).all() and (b[:,2]<a[:,2]).all()

def test_unprojected_inverted_primary_is_rejected():
 m=model()
 with torch.no_grad():m.power_gain.fill_(-1)
 with pytest.raises(ValueError):m(torch.full((1,6),.4))
