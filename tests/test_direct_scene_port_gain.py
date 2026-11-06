import pytest
import torch
from flybrain_drive.direct_scene_port_gain import DirectScenePortGain

def test_zero_preserves_original_values_exactly():
 torch.manual_seed(51);x=torch.randn(1024,42);gain=DirectScenePortGain(torch.randn(42));assert torch.equal(gain(x),x)

def test_reference_remains_fixed_and_input_gradient_is_preserved():
 ref=torch.linspace(-1,1,42);gain=DirectScenePortGain(ref);gain.delta.data.fill_(.05);x=ref[None].clone().requires_grad_();y=gain(x);assert torch.equal(y,x);y.sum().backward();torch.testing.assert_close(x.grad,torch.full_like(x,1.025))

def test_unbounded_or_invalid_inputs_rejected():
 gain=DirectScenePortGain(torch.zeros(42))
 with pytest.raises(ValueError):gain(torch.zeros(1,21))
 gain.delta.data[0]=.3
 with pytest.raises(ValueError):gain(torch.zeros(1,42))

def test_visual_coefficients_receive_gradient_at_zero():
 gain=DirectScenePortGain(torch.zeros(42));gain(torch.ones(1,42)).sum().backward();assert torch.equal(gain.delta.grad,torch.full((42,),.5))
