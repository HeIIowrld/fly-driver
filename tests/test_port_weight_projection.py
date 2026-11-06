import torch
from scripts.visual_native_port_experiment import bounded_step

def test_step_and_total_bounds_with_entry_box():
 torch.manual_seed(19);origin=torch.randn(12,46)*.01;p=torch.nn.Parameter(origin.clone())
 for _ in range(12):
  old=p.detach().clone()
  with torch.no_grad():p.add_(torch.randn_like(p)*10)
  bounded_step(p,old,origin,step_norm=.00025,total_norm=.001,entry_delta=.0002);assert (p-old).norm()<=.0002501 and (p-origin).norm()<=.0010001 and (p-origin).abs().max()<=.0002001

def test_native_current_global_lipschitz_bound():
 torch.manual_seed(37);origin=torch.randn(4,46)*.01;old=origin.clone();p=torch.nn.Parameter(origin+torch.randn_like(origin));bounded_step(p,old,origin,step_norm=.00025,total_norm=.01,entry_delta=.002);x=torch.cat([torch.full((20,19),5.),-torch.ones(20,19),torch.full((20,8),25.)],1);difference=(.25*torch.tanh(x@p.T)-.25*torch.tanh(x@old.T)).abs().max();bound=.25*(5494**.5)*.00025;assert difference<=bound+1e-7 and bound<.005

def test_visual_projection_preserves_previous_feasible_point_bound():
 origin=torch.full((4,320),.04);old=origin.clone();p=torch.nn.Parameter(origin+1.);bounded_step(p,old,origin,step_norm=.001,total_norm=.1,absolute=.08);assert (p-old).norm()<=.0010001 and p.abs().max()<=.08
