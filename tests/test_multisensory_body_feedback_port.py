import pytest
import torch
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort


def model():return MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),3)


def test_zero_initialization_preserves_original_input_current_and_absence_is_neutral():
 m=model();v=torch.randn(5,19);valid=torch.rand(5,19)>.2
 assert torch.equal(m(v,valid),torch.zeros(5,3))
 with torch.no_grad():m.projection.weight.normal_()
 assert torch.equal(m(v,torch.zeros_like(valid)),torch.zeros(5,3))


def test_temperature_slip_interaction_requires_both_real_signals():
 m=model();v=torch.zeros(1,19);valid=torch.ones(1,19,dtype=torch.bool)
 with torch.no_grad():m.projection.weight[0,38]=1
 v[0,3]=2;v[0,11]=1;positive=m(v,valid)
 v[0,11]=-1;negative=m(v,valid)
 assert positive[0,0]>0 and negative[0,0]<0
 valid[0,11]=False;v[0,11]=100
 assert m(v,valid)[0,0]==0
 assert (positive.abs()<=.25).all()


def test_invalid_values_or_implicit_masks_are_rejected():
 m=model();v=torch.zeros(1,19)
 with pytest.raises(ValueError):m(v,torch.ones_like(v))
 v[0,0]=float('nan')
 with pytest.raises(ValueError):m(v,torch.ones_like(v,dtype=torch.bool))
