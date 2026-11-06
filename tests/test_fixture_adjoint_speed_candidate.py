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
from flybrain_drive.symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand
import flybrain_drive.fixture_adjoint_speed_candidate as c

def fixture(tmp_path,monkeypatch):
 model=torch.nn.Module();model.speed_command_boundary=None;body=MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),3);parts=(model,*[torch.nn.Identity() for _ in range(3)],body,object());cal=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]));motion=SimpleNamespace(reference=torch.zeros(2048),scale=torch.ones(2048));current=SpatialMotionDNOverlay(FixedFullRoutingCurrent(cal,1.),motion.reference,motion.scale);policy=SymmetricNormalizedFlightSpeedCommand(torch.zeros(6),torch.ones(6));paths={}
 for key in ['speed_parent_checkpoint','speed_parent_optimizer','normalized_parent_checkpoint','normalized_parent_optimizer','bounded_parent_checkpoint','bounded_parent_optimizer','adjoint_parent_checkpoint','adjoint_parent_optimizer','motion_parent_checkpoint','parent_checkpoint','parent_port_checkpoint','fixed_decoder']:
  f=tmp_path/(key+'.pt')
  if key=='motion_parent_checkpoint':torch.save(dict(body_port=body.state_dict()),f)
  else:f.write_bytes(key.encode())
  paths[key]=str(f);paths[key+'_sha256']=sha256(f)
 n=dict(reference=[0.]*6,scale=[1.]*6);(tmp_path/'normalization-01.json').write_text(json.dumps(n));plan=dict(paths,maximum_power_input_sensitivity=2000.,speed_normalization=n,normalization_sha256=sha256(tmp_path/'normalization-01.json'),initial_gate=1.,minimum_gate=1.,visual_total_norm=2.,native_total_norm=.05,native_entry_delta=.02,speed_max_delta_m_s=10.,speed_weight_total_norm=.5,speed_weight_absolute_bound=.5,speed_bias_total_norm=.5,speed_bias_absolute_bound=.5);metadata=['fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','motion_parent_checkpoint_sha256','speed_parent_checkpoint_sha256','speed_parent_optimizer_sha256','normalized_parent_checkpoint_sha256','normalization_sha256','bounded_parent_checkpoint_sha256','maximum_power_input_sensitivity','adjoint_parent_checkpoint_sha256','adjoint_parent_optimizer_sha256','initial_gate','minimum_gate','speed_max_delta_m_s'];cp=dict(schema='flybrain.fixture_adjoint_speed_learning.experimental.v1',arm='input_adjoint',fixture_input_adjoint=True,current_calibration={k:v.clone() for k,v in current.state_dict().items()},body_port={k:v.clone() for k,v in body.state_dict().items()},speed_policy=policy.state_dict(),actor_qualified=False,**{k:plan[k] for k in metadata});directory=tmp_path/'fit-01';directory.mkdir();path=directory/'ports.pt';(directory/'manifest.json').write_text(json.dumps({'config':dict(plan=plan,arm='input_adjoint',fixture_input_adjoint=True)}));monkeypatch.setattr(c,'apply_power_bounded_speed',lambda path,parts,base,motion:(parts,current,{}));return path,cp,parts,cal,motion

def save(path,cp):
 torch.save(cp,path);(path.parent/'result.json').write_text(json.dumps(dict(status='completed_fixture_adjoint_speed_fit',checkpoint_sha256=sha256(path))))

def test_valid_policy_attaches_only_after_validation_and_has_no_authority(tmp_path,monkeypatch):
 path,cp,parts,cal,motion=fixture(tmp_path,monkeypatch);cp['speed_policy']['projection.weight'][0,-1]=-.01;cp['speed_policy']['projection.bias'][0]=.02;cp['body_port']['projection.weight'][0,0]=.001;save(path,cp);changed,current,r=c.apply_fixture_adjoint_speed(path,parts,cal,motion);assert changed[0].speed_command_boundary is not None and r['actor_qualified'] is False and r['vehicle_control_sent'] is False;assert parts[0].speed_command_boundary is None and changed[0] is not parts[0];assert parts[4].projection.weight[0,0]==0 and changed[4] is not parts[4];assert not any(p.requires_grad for p in changed[0].speed_command_boundary.parameters())

@pytest.mark.parametrize('bad',['parent','motion_norm','speed_dtype','positive_speed_gain','speed_norm','native_entry','authority','hash','reference','scale','normalization_hash','power_sensitivity','fixture_flag'])
def test_reject_without_attaching_policy_or_mutating_body(tmp_path,monkeypatch,bad):
 path,cp,parts,cal,motion=fixture(tmp_path,monkeypatch);before={k:v.clone() for k,v in parts[4].state_dict().items()}
 if bad=='parent':cp['current_calibration']['parent.registration.projection.weight'][0,0]=.01
 if bad=='motion_norm':cp['current_calibration']['motion_scale'][0]=2
 if bad=='speed_dtype':cp['speed_policy']['projection.weight']=cp['speed_policy']['projection.weight'].double()
 if bad=='positive_speed_gain':cp['speed_policy']['projection.weight'][0,-1]=.01
 if bad=='speed_norm':cp['speed_policy']['projection.weight'][0,:6]=.3
 if bad=='native_entry':cp['body_port']['projection.weight'][0,0]=.021
 if bad=='power_sensitivity':
  cp['speed_policy']['projection.weight'][0,2]=.01;cp['maximum_power_input_sensitivity']=.001
  metadata=json.loads((path.parent/'manifest.json').read_text());metadata['config']['plan']['maximum_power_input_sensitivity']=.001;(path.parent/'manifest.json').write_text(json.dumps(metadata))
 if bad=='fixture_flag':cp['fixture_input_adjoint']=False
 if bad=='reference':cp['speed_policy']['reference'][0]=.01
 if bad=='scale':cp['speed_policy']['scale'][0]=2
 if bad=='normalization_hash':cp['normalization_sha256']='bad'
 if bad=='authority':cp['actor_qualified']=True
 save(path,cp)
 if bad=='hash':path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError,match='Declared bounded direct power sensitivity' if bad=='power_sensitivity' else None):c.apply_fixture_adjoint_speed(path,parts,cal,motion)
 assert parts[0].speed_command_boundary is None and all(torch.equal(parts[4].state_dict()[k],v) for k,v in before.items())
