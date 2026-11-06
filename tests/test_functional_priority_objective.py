from types import SimpleNamespace
import pytest
import torch
from flybrain_drive.functional_priority_objective import gradient_snapshot,gradient_restore,recording_losses,split_functional_loss,DYNAMIC
from flybrain_drive.functional_bend_objective import functional_bend_loss

def test_shared_prefix_preserves_gradient_across_two_branches_and_clocks():
    parameter=torch.tensor(2.,requires_grad=True);model=SimpleNamespace(**{key:None for key in DYNAMIC});model.state=parameter.square();model.traces=(parameter*3,);model.previous_time=100.;model.previous_render_lowers=(99.,99.001,99.002,99.003);state=gradient_snapshot(model);gradient_restore(model,state);first=model.state*3;gradient_restore(model,state);second=model.state*5;(first+second).backward();assert parameter.grad==32 and model.previous_time==100. and model.previous_render_lowers==state['previous_render_lowers'];assert model.state.data_ptr()!=state['state'].data_ptr()

def test_incomplete_prefix_rejected():
    with pytest.raises(ValueError):gradient_restore(SimpleNamespace(),{'state':None})

def test_protection_does_not_reward_teacher_fit_inside_allowed_region():
    prediction=torch.full((4,3),.4,requires_grad=True);anchor=torch.full((4,3),.5);target=torch.zeros(4,3);mask=torch.ones(4,dtype=torch.bool);supervision,preservation,protection=recording_losses(prediction,anchor,target,mask,mask,anchor_weight=.5,maximum_ratio=1.02,protection_weight=1.);assert supervision>0 and preservation>0 and protection==0;assert torch.equal(torch.autograd.grad(protection,prediction)[0],torch.zeros_like(prediction))

def test_three_axis_worsening_activates_only_preservation_constraint():
    prediction=torch.tensor([[.5,.8,.5]],requires_grad=True);anchor=torch.full((1,3),.5);target=torch.zeros(1,3);mask=torch.ones(1,dtype=torch.bool);_,_,loss=recording_losses(prediction,anchor,target,mask,mask,anchor_weight=.5,maximum_ratio=1.02,protection_weight=1.);gradient=torch.autograd.grad(loss,prediction)[0];assert loss>0 and gradient[0,1]>0 and gradient[0,0]==0 and gradient[0,2]==0

def test_unlabelled_pre_frames_do_not_become_teacher_constraints():
    prediction=torch.tensor([[.5,.5,.5],[.8,.8,.8]],requires_grad=True);anchor=torch.full((2,3),.5);target=torch.zeros(2,3);fit=torch.tensor([True,False]);pre=torch.tensor([False,True]);_,_,loss=recording_losses(prediction,anchor,target,fit,pre,anchor_weight=.5,maximum_ratio=1.02,protection_weight=1.);assert loss==0

def test_split_functional_objective_matches_existing_numeric_loss_and_gradient():
    right=torch.tensor([[.1,.3,.2]],requires_grad=True);left=torch.tensor([[.11,.31,.21]],requires_grad=True);anchor=torch.tensor([[[.09,.3,.2]],[[.09,.3,.2]]]);args=dict(left_sign=-1,target_margin=.002,direction_weight=1000.,anchor_weight=.5);pieces=split_functional_loss(right,left,anchor,**args);old=functional_bend_loss(right,left,anchor[0],anchor[1],**args);assert torch.equal(sum(pieces),old);newgrad=torch.autograd.grad(sum(pieces),(right,left),retain_graph=True);oldgrad=torch.autograd.grad(old,(right,left));assert all(torch.equal(a,b) for a,b in zip(newgrad,oldgrad))
