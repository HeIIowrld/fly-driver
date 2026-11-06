import pytest
import torch
from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.spatial_motion_dn_overlay import SpatialMotionDNOverlay


def make():
 c=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]));parent=FixedFullRoutingCurrent(c,1.)
 with torch.no_grad():parent.registration.projection.weight.fill_(.002)
 return SpatialMotionDNOverlay(parent,torch.zeros(2048),torch.ones(2048))


def test_zero_overlay_exactly_preserves_nontrivial_parent():
 m=make();s=torch.linspace(-1,1,640).reshape(2,320);x=torch.randn(2,2048);p=torch.tensor([[0.,.2,.01,.17]]).repeat(2,1)
 assert torch.equal(m(s,x,p),m.parent(s,p))


def test_motion_can_drive_both_current_directions_without_changing_parent():
 m=make();s=torch.zeros(1,320);p=torch.full((1,4),.15);prior={k:v.clone() for k,v in m.parent.state_dict().items()}
 with torch.no_grad():m.motion_projection.weight[:,0]=torch.tensor([1.,-1.,1.,-1.])
 y=m(s,torch.ones(1,2048),p);assert y[0,0]>.1 and y[0,1]<.1 and (y>=0).all() and (y<=.2).all()
 y.sum().backward();assert m.motion_projection.weight.grad.norm()>0 and all(p.grad is None and not p.requires_grad for p in m.parent.parameters());assert all(torch.equal(m.parent.state_dict()[k],v) for k,v in prior.items())


@pytest.mark.parametrize('bad',[None,torch.zeros(1,512),torch.full((1,2048),float('nan')),torch.zeros(1,2048,dtype=torch.float64)])
def test_missing_invalid_motion_is_explicit(bad):
 with pytest.raises(ValueError):make()(torch.zeros(1,320),bad,torch.zeros(1,4))


def test_changed_parent_gate_rejected():
 m=make()
 with torch.no_grad():m.parent.registration.gate[0]=.5
 with pytest.raises(ValueError):m(torch.zeros(1,320),torch.zeros(1,2048),torch.zeros(1,4))
