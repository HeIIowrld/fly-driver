import itertools
import torch
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.complete_wheel_load_feedback import CompleteWheelLoadFeedback

def test_full_valid_exact_and_partial_load_contribution_removed():
    torch.manual_seed(712);base=CenteredSpeedBodyFeedbackAdapter(5)
    with torch.no_grad():base.projection.weight.normal_()
    gate=CompleteWheelLoadFeedback(base);x=torch.randn(1,21);x[:,8:12]=torch.tensor([.1,.2,.3,.4]);mask=torch.ones(1,21,dtype=torch.bool)
    assert torch.equal(gate(x,mask),base(x,mask))
    for pattern in itertools.product((False,True),repeat=4):
        if all(pattern):continue
        m=mask.clone();m[:,8:12]=torch.tensor(pattern)
        y=x.clone();y[:,8:12]=0
        assert torch.equal(gate(x,m),base(y,m))

def test_no_other_body_inputs_removed():
    torch.manual_seed(713);base=CenteredSpeedBodyFeedbackAdapter(7)
    with torch.no_grad():base.projection.weight.normal_()
    gate=CompleteWheelLoadFeedback(base);x=torch.randn(2,21);m=torch.ones(2,21,dtype=torch.bool);m[0,8]=False
    y=x.clone();y[0,8:12]=0
    assert torch.equal(gate(x,m),base(y,m))
