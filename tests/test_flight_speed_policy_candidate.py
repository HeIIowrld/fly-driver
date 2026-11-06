"""Synthetic checkpoint contracts, no measured neural graph or actuators."""
import json
from types import SimpleNamespace
import pytest
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
from flybrain_drive.spatial_motion_dn_overlay import SpatialMotionDNOverlay
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
import flybrain_drive.flight_speed_policy_candidate as c

def fixture(tmp_path,monkeypatch):
 model=torch.nn.Module();model.speed_command_boundary=None;body=MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),3);parts=(model,*[torch.nn.Identity() for _ in range(3)],body,object());cal=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]));motion=SimpleNamespace(reference=torch.zeros(2048),scale=torch.ones(2048));current=SpatialMotionDNOverlay(FixedFullRoutingCurrent(cal,1.),motion.reference,motion.scale);policy=FlightStateSpeedCommand(6);paths={}
 for key in ['speed_parent_checkpoint','speed_parent_optimizer','motion_parent_checkpoint','parent_checkpoint','parent_port_checkpoint','fixed_decoder']:
  f=tmp_path/(key+'.pt')
  if key=='motion_parent_checkpoint':torch.save(dict(body_port=body.state_dict()),f)
  else:f.write_bytes(key.encode())
  paths[key]=str(f);paths[key+'_sha256']=sha256(f)
 plan=dict(paths,initial_gate=1.,minimum_gate=1.,visual_total_norm=2.,native_total_norm=.05,native_entry_delta=.02,speed_max_delta_m_s=10.,speed_weight_total_norm=.5,speed_weight_absolute_bound=.5,speed_bias_total_norm=.5,speed_bias_absolute_bound=.5);metadata=['fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','motion_parent_checkpoint_sha256','speed_parent_checkpoint_sha256','speed_parent_optimizer_sha256','initial_gate','minimum_gate','speed_max_delta_m_s'];cp=dict(schema='flybrain.flight_speed_policy_learning.experimental.v1',arm='speed_policy',current_calibration={k:v.clone() for k,v in current.state_dict().items()},body_port={k:v.clone() for k,v in body.state_dict().items()},speed_policy=policy.state_dict(),actor_qualified=False,**{k:plan[k] for k in metadata});path=tmp_path/'ports.pt';(tmp_path/'manifest.json').write_text(json.dumps({'config':dict(plan=plan,arm='speed_policy')}));monkeypatch.setattr(c,'apply_spatial_motion_dn',lambda path,parts,base,motion:(parts,current,{}));return path,cp,parts,cal,motion

def save(path,cp):
 torch.save(cp,path);(path.parent/'result.json').write_text(json.dumps(dict(status='completed_flight_speed_policy_fit',checkpoint_sha256=sha256(path))))

def test_valid_policy_attaches_only_after_validation_and_has_no_authority(tmp_path,monkeypatch):
 path,cp,parts,cal,motion=fixture(tmp_path,monkeypatch);cp['speed_policy']['projection.weight'][0,-1]=-.01;cp['speed_policy']['projection.bias'][0]=.02;cp['body_port']['projection.weight'][0,0]=.001;save(path,cp);changed,current,r=c.apply_flight_speed_policy(path,parts,cal,motion);assert changed[0].speed_command_boundary is not None and r['actor_qualified'] is False and r['vehicle_control_sent'] is False;assert parts[4].projection.weight[0,0]==0 and changed[4] is not parts[4];assert not any(p.requires_grad for p in changed[0].speed_command_boundary.parameters())

@pytest.mark.parametrize('bad',['parent','motion_norm','speed_dtype','positive_speed_gain','speed_norm','native_entry','authority','hash'])
def test_reject_without_attaching_policy_or_mutating_body(tmp_path,monkeypatch,bad):
 path,cp,parts,cal,motion=fixture(tmp_path,monkeypatch);before={k:v.clone() for k,v in parts[4].state_dict().items()}
 if bad=='parent':cp['current_calibration']['parent.registration.projection.weight'][0,0]=.01
 if bad=='motion_norm':cp['current_calibration']['motion_scale'][0]=2
 if bad=='speed_dtype':cp['speed_policy']['projection.weight']=cp['speed_policy']['projection.weight'].double()
 if bad=='positive_speed_gain':cp['speed_policy']['projection.weight'][0,-1]=.01
 if bad=='speed_norm':cp['speed_policy']['projection.weight'][0,:6]=.3
 if bad=='native_entry':cp['body_port']['projection.weight'][0,0]=.021
 if bad=='authority':cp['actor_qualified']=True
 save(path,cp)
 if bad=='hash':path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError):c.apply_flight_speed_policy(path,parts,cal,motion)
 assert parts[0].speed_command_boundary is None and all(torch.equal(parts[4].state_dict()[k],v) for k,v in before.items())
