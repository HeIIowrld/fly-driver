import pytest
import torch
from flybrain_drive.visual_anticipatory_objective import visual_anticipatory_loss
from scripts.native_visual_experiment import tensor_state_equal


def test_auxiliary_target_only_supplies_gradient_to_muscle_forecast():
    prediction=torch.tensor([[.1,.2,.3]],requires_grad=True)
    target=torch.tensor([.1,.2,.3],requires_grad=True)
    source=torch.tensor([.2,.3],requires_grad=True)
    forecast=torch.tensor([.05],requires_grad=True)
    future=torch.tensor(.25,requires_grad=True)
    loss=visual_anticipatory_loss(prediction,target,source,pre=True,forecast=forecast,future_target=future)
    loss.backward()
    assert forecast.grad.item()<0
    assert target.grad is None and source.grad is None and future.grad is None
    torch.testing.assert_close(prediction.grad,torch.zeros_like(prediction))


def test_present_arm_has_current_command_and_pedal_anchor_gradients():
    prediction=torch.tensor([[.1,.4,.5]],requires_grad=True)
    loss=visual_anticipatory_loss(prediction,torch.tensor([.2,.2,.3]),torch.tensor([.2,.3]),pre=False)
    loss.backward()
    assert prediction.grad[0,0]<0
    assert torch.all(prediction.grad[0,1:]>0)


def test_future_loss_outside_pre_window_is_rejected():
    with pytest.raises(ValueError):
        visual_anticipatory_loss(torch.zeros(1,3),torch.zeros(3),torch.zeros(2),pre=False,
            forecast=torch.zeros(1),future_target=torch.ones(1))


def test_partial_future_supervision_is_rejected():
    with pytest.raises(ValueError):
        visual_anticipatory_loss(torch.zeros(1,3),torch.zeros(3),torch.zeros(2),pre=True,forecast=torch.zeros(1))


def test_sparse_flight_state_requires_identical_edges_and_values():
    edges=torch.tensor([[0,1],[1,0]])
    first={'graph':torch.sparse_coo_tensor(edges,torch.tensor([1.,2.]),(2,2)).coalesce()}
    same={'graph':first['graph'].clone()}
    changed={'graph':torch.sparse_coo_tensor(edges,torch.tensor([1.,3.]),(2,2)).coalesce()}
    assert tensor_state_equal(first,same)
    assert not tensor_state_equal(first,changed)
