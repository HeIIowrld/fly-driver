import torch
import pytest
from flybrain_drive.visual_port_current_correction import VisualPortCurrentCorrection


def module():return VisualPortCurrentCorrection(torch.zeros(320),torch.ones(320))


def test_zero_correction_preserves_original_current_bytes_including_bounds():
    m=module();x=torch.randn(3,320);base=torch.tensor([[0.,.001,.18,.2]]).repeat(3,1)
    assert torch.equal(m(x,base),base)


def test_four_port_gradients_are_independent_and_never_bypass_current_bounds():
    m=module();x=torch.ones(1,320);base=torch.full((1,4),.1)
    m(x,base)[0,0].backward()
    assert torch.count_nonzero(m.projection.weight.grad[0])==320
    assert torch.count_nonzero(m.projection.weight.grad[1:])==0
    with torch.no_grad():m.projection.weight[0,0]=100
    out=m(x,base)
    assert out[0,0]>.1 and torch.equal(out[:,1:],base[:,1:])
    assert (out>=0).all() and (out<=.2).all()


def test_nonfinite_features_or_invalid_current_contract_are_rejected():
    m=module();x=torch.ones(1,320);base=torch.full((1,4),.1)
    x[0,0]=float('nan')
    with pytest.raises(ValueError):m(x,base)
    with pytest.raises(ValueError):m(torch.zeros(1,320),torch.full((1,4),-.1))
