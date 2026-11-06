import pytest
import torch
from flybrain_drive.muscle_interaction_vehicle_readout import MuscleInteractionVehicleReadout,PAIRS

def parent():
 return dict(training_reference=torch.full((6,),.5),training_scale=torch.full((6,),.1),**{'projection.weight':torch.tensor([[.1,-.2,.3,-.2,.1,.1],[0.,0.,0.,0.,.1,.1],[0.,0.,0.,0.,-.1,-.1]]),'projection.bias':torch.tensor([.01,.1,-.1])})
def make(degree):
 n=21 if degree==2 else 0;return MuscleInteractionVehicleReadout(parent(),degree=degree,quadratic_reference=torch.zeros(n,dtype=torch.float64),quadratic_scale=torch.ones(n,dtype=torch.float64))

@pytest.mark.parametrize('degree',[1,2])
def test_baseline_steering_equivalence_and_exact_pedals(degree):
 m=make(degree);x=torch.linspace(.1,.9,120).reshape(20,6);p=m(x);b=m.base(x);assert p.dtype==torch.float32 and torch.equal(p[:,1:],b[:,1:]);assert abs(p[:,0]-b[:,0]).max()<2e-7

def test_cross_proxy_interaction_changes_slope():
 m=make(2)
 with torch.no_grad():m.steering.weight.zero_();m.steering.bias.zero_();m.steering.weight[0,6+PAIRS.index((0,1))]=.1
 x=torch.full((2,6),.5);x[:,0]=.6;x[1,1]=.6;y=m(x);assert y[0,0]==0 and .09<y[1,0]<.11

@pytest.mark.parametrize('bad',[torch.full((1,7),.5),torch.full((1,6),float('nan')),torch.full((1,6),1.1),torch.ones(1,6,dtype=torch.float64)])
def test_invalid_or_extra_actor_inputs_rejected(bad):
 with pytest.raises(ValueError):make(2)(bad)

def test_power_pedals_remain_monotone_with_nonlinear_steering():
 m=make(2)
 with torch.no_grad():m.steering.weight.fill_(.1)
 x=torch.full((1,6),.5);y=m(x);x[:,4:]+=.01;v=m(x);assert v[0,1]>y[0,1] and v[0,2]<y[0,2]

def test_invalid_normalization_rejected():
 with pytest.raises(ValueError):MuscleInteractionVehicleReadout(parent(),degree=2,quadratic_reference=torch.zeros(21,dtype=torch.float64),quadratic_scale=torch.zeros(21,dtype=torch.float64))
