from types import SimpleNamespace
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
from scripts.spatial_motion_dn_experiment import MotionDNCurrent
from scripts.symmetric_speed_experiment import SymmetricSpeedExperiment

def make():
 e=object.__new__(SymmetricSpeedExperiment);e.train_body=True;e.learn_speed=True;base=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]));e.calibration=MotionDNCurrent(FixedFullRoutingCurrent(base,1.),torch.zeros(2048),torch.ones(2048));e.body_port=SimpleNamespace(projection=torch.nn.Linear(46,3,bias=False));e.speed_policy=FlightStateSpeedCommand(6);e.current_origin=torch.zeros_like(e.calibration.projection.weight);e.native_origin=e.body_port.projection.weight.detach().clone();e.port_plan=dict(visual_step_norm=.0015,visual_total_norm=2.,native_step_norm=.00025,native_total_norm=.05,native_entry_delta=.02,visual_learning_rate=.001,native_learning_rate=.001,speed_weight_learning_rate=.001,speed_bias_learning_rate=.001,speed_weight_step_norm=.001,speed_weight_total_norm=.5,speed_weight_absolute_bound=.5,speed_bias_step_norm=.001,speed_bias_total_norm=.5,speed_bias_absolute_bound=.5);return e

def test_projected_speed_gain_cannot_reinforce_faster_sensed_speed():
 e=make();parameters=e.train_parameters();old=[p.detach().clone() for p in parameters]
 with torch.no_grad():parameters[2].add_(.1);parameters[3].add_(.1)
 parameters[2].grad=torch.ones_like(parameters[2]);parameters[3].grad=torch.ones_like(parameters[3]);e.project_after_step(old);assert e.speed_policy.projection.weight[0,-1]<=0 and float(parameters[2].detach().norm())<=.0010001 and float(parameters[3].detach().norm())<=.0010001;assert torch.equal(parameters[1],old[1])
 with torch.no_grad():e.speed_policy.projection.weight[0,-1]=-.1
 errors=[]
 for speed in [13.,26.,60.]:
  s=torch.tensor([[speed]]);v=e.speed_policy(torch.full((1,6),.5),s,torch.tensor([[0.,3.]]));errors.append(float((v[0,1]-speed).detach()))
 assert errors[0]>=errors[1]>=errors[2]

def test_real_adam_moments_are_loaded_before_new_policy_groups(tmp_path):
 e=make();parameters=e.train_parameters();old=torch.optim.Adam([dict(params=[p],lr=.001) for p in parameters[:2]])
 for j in range(2):
  parameters[0].grad=torch.ones_like(parameters[0]);parameters[1].grad=torch.ones_like(parameters[1]) if j==0 else None;old.step()
 path=tmp_path/'adam.pt';torch.save(old.state_dict(),path);e.port_plan.update(normalized_parent_optimizer=str(path),normalized_parent_optimizer_sha256=sha256(path),initial_optimizer_steps=[2,1]);new=e.make_optimizer();assert len(new.param_groups)==4 and int(new.state[parameters[0]]['step'])==2 and int(new.state[parameters[1]]['step'])==1;assert parameters[2] not in new.state and parameters[3] not in new.state;assert torch.equal(new.state[parameters[0]]['exp_avg'],old.state[parameters[0]]['exp_avg'])
