import pytest
import torch
from flybrain_drive.temporal_port_gain import TemporalPortGain,attach_temporal_gain

class Source:
    def __init__(self):self.calls=0
    def step(self,delta,traces):self.calls+=1;return delta,traces

def test_identity_and_removal_preserve_source_trace_objects():
    source=Source();gain=TemporalPortGain(3);delta=torch.tensor([[.2,-.3,.1]]);traces=(torch.ones(1,2),torch.zeros(1,2));handle=attach_temporal_gain(source,gain);actual,seen=source.step(delta,traces);assert torch.equal(actual,delta) and seen is traces and source.calls==1;handle.remove();handle.remove();actual,seen=source.step(delta,traces);assert actual is delta and seen is traces and source.calls==2

def test_gain_training_does_not_modify_source_or_traces():
    source=Source();gain=TemporalPortGain(2);handle=attach_temporal_gain(source,gain);delta=torch.tensor([[.2,-.3]]);traces=(torch.ones(1),);actual,seen=source.step(delta,traces);actual.sum().backward();assert torch.equal(gain.value.grad,delta[0]) and torch.equal(delta,torch.tensor([[.2,-.3]])) and seen is traces;handle.remove()

def test_projection_keeps_sign_and_positive_bounded_gain():
    gain=TemporalPortGain(2)
    with torch.no_grad():gain.value.copy_(torch.tensor([-1.,3.]))
    with pytest.raises(ValueError):gain(torch.ones(1,2))
    gain.project();assert torch.equal(gain.value,torch.tensor([.25,1.75]));assert torch.equal(gain(torch.tensor([[-1.,1.]])),torch.tensor([[-.25,1.75]]))

def test_double_attachment_rejected_then_explicitly_reusable():
    source=Source();gain=TemporalPortGain(1);handle=attach_temporal_gain(source,gain)
    with pytest.raises(ValueError):attach_temporal_gain(source,gain)
    handle.remove();new=attach_temporal_gain(source,gain);new.remove()

@pytest.mark.parametrize('bad',[torch.zeros(2,2),torch.ones(1,2,dtype=torch.float64),torch.tensor([[float('nan'),0.]])])
def test_invalid_delta_rejected(bad):
    with pytest.raises(ValueError):TemporalPortGain(2)(bad)
