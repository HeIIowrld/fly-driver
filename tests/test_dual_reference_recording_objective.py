import torch
from flybrain_drive.functional_priority_objective import recording_losses
from flybrain_drive.dual_reference_recording_objective import dual_reference_recording_losses
OPTIONS=dict(anchor_weight=1.,maximum_ratio=1.02,protection_weight=1.)

def test_equal_parent_reference_preserves_original_objective_exactly():
 torch.manual_seed(58);p=torch.randn(32,3);a=torch.randn(32,3);t=torch.randn(32,3);mask=torch.arange(32)%2==0;pre=torch.arange(32)<5
 old=recording_losses(p,a,t,mask,pre,**OPTIONS);new=dual_reference_recording_losses(p,a,a,t,mask,pre,**OPTIONS)
 assert all(torch.equal(x,y) for x,y in zip(old,new))

def test_stricter_reference_is_selected_separately_per_axis():
 p=torch.ones(4,3);old=torch.tensor([[2.,.5,2.]]).repeat(4,1);parent=torch.tensor([[.5,2.,2.]]).repeat(4,1);target=torch.zeros(4,3);fit=torch.ones(4,dtype=torch.bool);pre=torch.zeros_like(fit)
 value=dual_reference_recording_losses(p,old,parent,target,fit,pre,**OPTIONS)[2];strict=torch.minimum(old,parent);expected=recording_losses(p,strict,target,fit,pre,**OPTIONS)[2]
 assert torch.equal(value,expected) and value>recording_losses(p,old,target,fit,pre,**OPTIONS)[2]

def test_parent_and_teacher_are_detached_and_prediction_gradient_flows():
 p=torch.ones(4,3,requires_grad=True);a=torch.full((4,3),2.,requires_grad=True);b=torch.full((4,3),.5,requires_grad=True);t=torch.zeros(4,3,requires_grad=True);fit=torch.ones(4,dtype=torch.bool)
 sum(dual_reference_recording_losses(p,a,b,t,fit,fit,**OPTIONS)).backward();assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0;assert all(x.grad is None for x in [a,b,t])

def test_empty_fit_preserves_connected_zero_protection():
 p=torch.ones(4,3,requires_grad=True);a=torch.ones(4,3);fit=torch.zeros(4,dtype=torch.bool);v=dual_reference_recording_losses(p,a,a,a,fit,fit,**OPTIONS)[2];v.backward();assert v==0 and torch.equal(p.grad,torch.zeros_like(p))

def test_equal_reference_preserves_prediction_gradient():
 torch.manual_seed(59);p=torch.randn(32,3,requires_grad=True);a=torch.randn(32,3);t=torch.randn(32,3);fit=torch.ones(32,dtype=torch.bool);pre=torch.arange(32)<10
 old=sum(recording_losses(p,a,t,fit,pre,**OPTIONS));g0=torch.autograd.grad(old,p)[0];new=sum(dual_reference_recording_losses(p,a,a,t,fit,pre,**OPTIONS));g1=torch.autograd.grad(new,p)[0];torch.testing.assert_close(g0,g1,rtol=1e-6,atol=1e-7)
