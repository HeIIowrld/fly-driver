import numpy as np
import pytest
import torch
from types import SimpleNamespace
from flybrain_drive.motion_spatial_residual_port import MotionSpatialResidualPort
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation,validate_motion_binding


def port():return MotionSpatialResidualPort(torch.zeros(2048),torch.ones(2048),3)


def test_zero_initialization_preserves_original_visual_drive_exactly():
    p=port();x=torch.randn(2,2048);assert torch.equal(p(x),torch.zeros(2,3))
    assert not p.reference.requires_grad and not p.scale.requires_grad


def test_selected_view_direction_and_spatial_cell_are_not_permuted():
    p=port();index=np.ravel_multi_index((2,5,3,6),(4,8,8,8))
    with torch.no_grad():p.projection.weight[1,index]=2.
    x=torch.zeros(1,2048);x[0,index]=1.;expected=torch.tensor([[0.,.25*torch.tanh(torch.tensor(2.)),0.]])
    torch.testing.assert_close(p(x),expected)
    x[0,index]=0;x[0,index+1]=1.;assert torch.equal(p(x),torch.zeros(1,3))


def test_motion_port_is_bounded_and_differentiable():
    p=port()
    with torch.no_grad():p.projection.weight.fill_(100.)
    assert float(p(torch.ones(1,2048)).detach().abs().max())<=.25
    p=port();x=torch.ones(1,2048);p(x).sum().backward();assert p.projection.weight.grad.abs().sum()>0


@pytest.mark.parametrize('x',[torch.zeros(1,256),torch.zeros(1,2048,dtype=torch.float64),torch.full((1,2048),float('nan'))])
def test_invalid_or_legacy_input_rejected(x):
    with pytest.raises(ValueError):port()(x)


def observation(**changes):
    fields=dict(episode_id='one',camera_receipt_time=1.,render_start_lowers=(.9,.91,.92,.93),coarse_motion_reference=np.zeros(512),features=np.zeros(2048));fields.update(changes);return MotionSpatialObservation(**fields)


def test_motion_observation_copies_and_freezes_arrays():
    x=np.ones(2048);o=observation(features=x);x[0]=2;assert o.features[0]==1
    with pytest.raises(ValueError):o.features[0]=3


@pytest.mark.parametrize('change',[dict(episode_id='other'),dict(camera_receipt_time=1.01),dict(render_start_lowers=(.8,.91,.92,.93)),dict(coarse_motion_reference=np.ones(512))])
def test_binding_rejects_another_scene_or_original_clock(change):
    context=SimpleNamespace(camera_receipt_time=1.,render_start_lowers=(.9,.91,.92,.93))
    with pytest.raises(ValueError):validate_motion_binding(observation(**change),episode_id='one',scene_features=np.zeros(576),body_context=context)
