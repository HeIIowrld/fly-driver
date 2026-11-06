import pytest
import torch
from flybrain_drive.full_range_dn_registration import FullRangeDNRegistration

def make():return FullRangeDNRegistration(torch.zeros(320),torch.ones(320))
def test_zero_gate_exact_parent():
 m=make();x=torch.linspace(-5,5,640).reshape(2,320);p=torch.tensor([[.13,1e-7,.13,1e-7],[0.,.2,.001,.19]]);assert torch.equal(m(x,p),p)
def test_can_reverse_closed_population_without_exceeding_current_contract():
 m=make();p=torch.tensor([[.13,1e-7,.13,1e-7]])
 with torch.no_grad():m.gate.fill_(1);m.projection.bias.copy_(torch.tensor([-2.,2.,-2.,2.]))
 v=m(torch.zeros(1,320),p);assert (v>=0).all() and (v<=.2).all() and v[0,[1,3]].mean()>v[0,[0,2]].mean()
def test_gradient_stages_are_explicit():
 m=make();x=torch.ones(1,320);p=torch.tensor([[.13,1e-7,.13,1e-7]]);m(x,p).square().sum().backward();assert m.gate.grad.abs().max()>0 and torch.count_nonzero(m.projection.weight.grad)==0;m.zero_grad(set_to_none=True)
 with torch.no_grad():m.gate.fill_(.1)
 m(x,p).square().sum().backward();assert m.projection.weight.grad.abs().max()>0
@pytest.mark.parametrize('bad',[float('nan'),-.1,.21])
def test_reject_invalid_parent_current(bad):
 with pytest.raises(ValueError):make()(torch.zeros(1,320),torch.full((1,4),bad))
def test_unprojected_gate_rejected():
 m=make()
 with torch.no_grad():m.gate.fill_(1.1)
 with pytest.raises(ValueError):m(torch.zeros(1,320),torch.zeros(1,4))
 m.project();assert (m.gate==1).all()
