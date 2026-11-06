import pytest
import torch
from flybrain_drive.fixture_speed_adjoint import frozen_parameter_speed_command
from flybrain_drive.symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand

def policy():
 p=SymmetricNormalizedFlightSpeedCommand(torch.tensor([.001,.0008,.0001,.002,.0014,.0014]),torch.tensor([.0007,.0004,.00001,.0002,.00013,.00004]))
 with torch.no_grad():p.projection.weight.copy_(torch.tensor([[.01,-.02,.001,-.004,-.003,-.001,-.002]]));p.projection.bias.fill_(.02)
 return p

def test_exact_output_and_input_gradient_with_no_parameter_gradient():
 p=policy();x=torch.tensor([[.001,.003,.0012,.002,.0013,.0015]],requires_grad=True);s=torch.tensor([[30.]]);g=torch.tensor([[0.,3.]])
 normal=p(x,s,g);frozen=frozen_parameter_speed_command(p,x,s,g);assert torch.equal(normal,frozen)
 expected=torch.autograd.grad(normal[0,1],x)[0];actual=torch.autograd.grad(frozen[0,1],(x,*p.parameters()),allow_unused=True);assert torch.equal(expected,actual[0]) and expected.abs().max()>0 and all(v is None for v in actual[1:])

def test_frozen_coefficients_still_allow_upstream_learning_at_episode_start():
 p=policy();x=torch.zeros(1,6,requires_grad=True);s=torch.tensor([[13.]]);g=torch.tensor([[0.,3.]])
 out=frozen_parameter_speed_command(p,x,s,g);assert torch.equal(out,p(x,s,g));out.sum().backward();assert x.grad is not None and all(v.grad is None for v in p.parameters())

def test_invalid_racing_goals_are_rejected():
 p=policy()
 with pytest.raises(ValueError):frozen_parameter_speed_command(p,torch.zeros(1,6),torch.tensor([[30.]]),torch.tensor([[1.,3.]]))
