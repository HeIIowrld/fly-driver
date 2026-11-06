import pytest
import torch
from flybrain_drive.symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand as Policy

def make():return Policy(torch.tensor([.001,.002,.0001,.002,.001,.001]),torch.tensor([.001,.001,.00001,.0001,.0001,.0001]))
def test_pair_swap_preserves_speed_target_and_gradient():
 p=make()
 with torch.no_grad():p.projection.weight.copy_(torch.tensor([[.1,-.1,.03,.02,.03,-.02,-.1]]))
 x=torch.tensor([[.001,.003,.002,.001,.0011,.0012]],requires_grad=True);speed=torch.tensor([[30.]]);goals=torch.tensor([[0.,3.]])
 a=p(x,speed,goals);b=p(x[:,[1,0,3,2,5,4]],speed,goals);assert torch.equal(a,b);a[0,1].backward();assert torch.isfinite(x.grad).all() and x.grad.abs().max()>0

def test_zero_coefficients_exactly_preserve_neutral_speed_and_clipping():
 p=make()
 for v in [0.,13.,74.,90.]:
  out=p(torch.zeros(1,6),torch.tensor([[v]]),torch.tensor([[0.,3.]]));assert out[0,0]==0 and out[0,1]==min(v,3.*(100./3.6))
 assert p.encode(torch.ones(1,6),torch.tensor([[30.]]))[0,:6].abs().max()<=5

@pytest.mark.parametrize('scale',[torch.zeros(6),torch.full((6,),float('nan')),torch.ones(5),torch.ones(6,dtype=torch.float64)])
def test_invalid_train_normalization_is_rejected(scale):
 with pytest.raises(ValueError):Policy(torch.zeros(6),scale)

def test_invalid_body_and_goal_are_not_repaired():
 p=make()
 with pytest.raises(ValueError):p(torch.zeros(1,6),torch.tensor([[-1.]]),torch.tensor([[0.,3.]]))
 with pytest.raises(ValueError):p(torch.zeros(1,6),torch.tensor([[30.]]),torch.tensor([[1.,3.]]))
 with pytest.raises(ValueError):p(torch.zeros(1,6,dtype=torch.float64),torch.tensor([[30.]]),torch.tensor([[0.,3.]]))
