import pytest
import torch
from flybrain_drive.joint_sensory_objective import joint_sensory_loss,project_sensory_weight


def test_targets_and_old_predictions_never_become_actor_gradient_paths():
    p=torch.ones(1,3,requires_grad=True);y=torch.zeros(3,requires_grad=True);anchor=torch.zeros(3,requires_grad=True)
    loss=joint_sensory_loss(p,y,anchor,pre=False);loss.backward()
    assert y.grad is None and anchor.grad is None
    torch.testing.assert_close(p.grad,torch.tensor([[12.,3.,3.]]))


def test_precorner_priority_adds_steering_and_braking_gradient():
    p=torch.ones(1,3,requires_grad=True);joint_sensory_loss(p,torch.zeros(3),torch.zeros(3),pre=True).backward()
    torch.testing.assert_close(p.grad,torch.tensor([[44.,3.,9.]]))


def test_projection_limits_change_without_replacing_reference_or_in_range_weight():
    reference=torch.tensor([1.,2.,3.]);weight=torch.nn.Parameter(torch.tensor([2.,1.,3.005]));project_sensory_weight(weight,reference,.01)
    torch.testing.assert_close(weight,torch.tensor([1.01,1.99,3.005]));torch.testing.assert_close(reference,torch.tensor([1.,2.,3.]))


@pytest.mark.parametrize('bound',[0.,.03,float('nan'),True])
def test_undefined_projection_bound_rejected(bound):
    with pytest.raises(ValueError):project_sensory_weight(torch.zeros(1),torch.zeros(1),bound)


def test_nonfinite_loss_or_projection_is_rejected():
    with pytest.raises(ValueError):joint_sensory_loss(torch.full((1,3),float('nan')),torch.zeros(3),torch.zeros(3),pre=False)
    with pytest.raises(ValueError):project_sensory_weight(torch.tensor([float('inf')]),torch.zeros(1),.01)
