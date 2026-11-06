import torch
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from scripts.full_range_port_experiment import RegisteredCurrent


def make():
    base=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]))
    with torch.no_grad():base.projection.weight.fill_(.001)
    return base,RegisteredCurrent(base,.01)


def test_frozen_base_is_copied_and_preserved_while_bilateral_registration_learns():
    base,m=make();x=torch.ones(1,320);p=torch.tensor([[.13,1e-7,.13,1e-7]])
    prior={k:v.clone() for k,v in base.state_dict().items()}
    m(x,p).sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in m.base.parameters())
    assert (m.projection.weight.grad.norm(dim=1)>0).all()
    opt=torch.optim.Adam([m.projection.weight,m.projection.bias,m.gate],lr=.001)
    opt.step();m.registration.project()
    assert all(torch.equal(m.base.state_dict()[k],v) for k,v in prior.items())
    assert all(torch.equal(base.state_dict()[k],v) for k,v in prior.items())


def test_zero_registration_gate_preserves_nonzero_parent_codec():
    base,m=make();x=torch.ones(2,320);p=torch.tensor([[.13,1e-7,.13,1e-7]]).repeat(2,1)
    with torch.no_grad():m.gate.zero_()
    assert torch.equal(m(x,p),base(x,p))
    assert not torch.equal(m(x,p),p)
