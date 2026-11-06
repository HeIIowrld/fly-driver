from types import SimpleNamespace
import torch
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
from scripts.spatial_motion_dn_experiment import SpatialMotionDNExperiment,MotionDNCurrent


def test_two_group_projection_keeps_static_parent_and_native_none_exact():
 e=object.__new__(SpatialMotionDNExperiment);e.train_body=True;base=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]));e.calibration=MotionDNCurrent(FixedFullRoutingCurrent(base,1.),torch.zeros(2048),torch.ones(2048));e.body_port=SimpleNamespace(projection=torch.nn.Linear(46,201,bias=False));e.current_origin=e.calibration.projection.weight.detach().clone();e.native_origin=e.body_port.projection.weight.detach().clone();e.port_plan=dict(visual_step_norm=.0015,visual_total_norm=2.,native_step_norm=.00025,native_total_norm=.05,native_entry_delta=.02);p=e.train_parameters();assert len(p)==2 and e.native_parameter_index==1 and p[0].numel()==8192;before=[w.detach().clone() for w in p];parent={k:v.clone() for k,v in e.calibration.parent.state_dict().items()}
 with torch.no_grad():p[0].add_(.1)
 assert p[1].grad is None;e.project_after_step(before)
 assert torch.equal(p[1],before[1]) and float((p[0].detach()-before[0]).norm())<=.0015001
 assert all(torch.equal(e.calibration.parent.state_dict()[k],v) for k,v in parent.items())
