import torch
from flybrain_drive.fixed_history_trajectory_gradient import fixed_history_trajectory_gradient
from flybrain_drive.differentiable_planar_vehicle import differentiable_bicycle_step

def test_two_pass_vjp_matches_full_vehicle_gradient():
    dtype=torch.float64;initial=torch.tensor([[12.,0.,0.,3.]],dtype=dtype);feature=torch.linspace(-.2,.3,20,dtype=dtype);p=torch.tensor(.4,dtype=dtype,requires_grad=True);control=p*feature;acc=torch.zeros_like(control)
    loss,g,_=fixed_history_trajectory_gradient(initial,control,acc,bend=1);surrogate=(p*feature*g).sum();surrogate.backward();surrogate_grad=p.grad.detach().clone();p.grad=None
    world=initial;cost=[]
    for steer in p*feature:
        world=differentiable_bicycle_step(world,steer.reshape(1)*.35,torch.zeros(1,dtype=dtype),dt_s=.04);u=(world[:,0]-12).clamp_min(0);center=.018*u.square();heading=torch.atan(.036*u);cost.append((((world[:,1]-center)/.3)**2+.25*((world[:,2]-heading)/.15)**2).mean())
    exact=torch.stack(cost).mean();exact.backward();assert torch.allclose(exact,loss,atol=1e-12,rtol=0);assert torch.allclose(p.grad,surrogate_grad,atol=1e-12,rtol=1e-9)
