import pytest
import torch
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.reference_load_feedback import ReferenceLoadFeedback

def test_reference_is_exact_and_changed_load_is_causal():
    torch.manual_seed(7);base=CenteredSpeedBodyFeedbackAdapter(3)
    with torch.no_grad():base.projection.weight.normal_(0,.1)
    weight=torch.tensor([[1.,1.,-1.,-1.],[-1.,-1.,1.,1.],[.5,-.5,.5,-.5]])
    reference=torch.tensor([.25]*4);candidate=ReferenceLoadFeedback(base,weight,reference)
    body=torch.zeros((1,21));body[:,8:12]=reference;valid=torch.ones_like(body,dtype=torch.bool)
    torch.testing.assert_close(candidate(body,valid),base(body,valid),rtol=0,atol=0)
    changed=body.clone();changed[:,8:12]+=torch.tensor([.025,.025,-.025,-.025])
    assert not torch.equal(candidate(changed,valid),base(changed,valid))

def test_missing_invalid_and_non_normalized_loads_fail():
    base=CenteredSpeedBodyFeedbackAdapter(3);candidate=ReferenceLoadFeedback(base,torch.ones(3,4),[.25]*4)
    body=torch.zeros((1,21));body[:,8:12]=.25;valid=torch.ones_like(body,dtype=torch.bool)
    valid[:,8]=False
    with pytest.raises(ValueError,match='four'):candidate(body,valid)
    valid[:,8]=True;body[:,8]=-.1
    with pytest.raises(ValueError,match='Normalized'):candidate(body,valid)
