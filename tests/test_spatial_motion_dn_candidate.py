"""Synthetic schema/sensory normalization checks, no biological graph or controls."""
import json
from types import SimpleNamespace
import pytest
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.spatial_motion_dn_overlay import SpatialMotionDNOverlay
import flybrain_drive.spatial_motion_dn_candidate as c


def fixture(tmp_path,monkeypatch):
 paths={}
 for key in ['motion_parent_checkpoint','parent_checkpoint','parent_port_checkpoint','fixed_decoder']:
  p=tmp_path/(key+'.pt');p.write_bytes(key.encode());paths[key]=str(p);paths[key+'_sha256']=sha256(p)
 cal=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]));body=MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),3);parts=tuple([torch.nn.Identity() for _ in range(4)]+[body,object()]);motion=SimpleNamespace(reference=torch.zeros(2048),scale=torch.ones(2048));monkeypatch.setattr(c,'apply_fixed_full_routing',lambda path,parts,base:(parts,FixedFullRoutingCurrent(base,1.),{}));current=SpatialMotionDNOverlay(FixedFullRoutingCurrent(cal,1.),motion.reference,motion.scale)
 plan=dict(paths,initial_gate=1.,minimum_gate=1.,visual_total_norm=2.,native_total_norm=.05,native_entry_delta=.02)
 cp=dict(schema='flybrain.spatial_motion_dn_learning.experimental.v1',arm='visual_native',current_calibration=current.state_dict(),body_port={k:v.clone() for k,v in body.state_dict().items()},actor_qualified=False,**{k:plan[k] for k in ['fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','motion_parent_checkpoint_sha256','initial_gate','minimum_gate']});path=tmp_path/'ports.pt';(tmp_path/'manifest.json').write_text(json.dumps({'config':dict(plan=plan,arm='visual_native')}));return path,cp,parts,cal,motion


def save(path,cp):
 torch.save(cp,path);(path.parent/'result.json').write_text(json.dumps(dict(status='completed_spatial_motion_dn_fit',checkpoint_sha256=sha256(path))))


def test_valid_overlay_preserves_callers_and_has_no_authority(tmp_path,monkeypatch):
 path,cp,parts,cal,motion=fixture(tmp_path,monkeypatch);cp['body_port']['projection.weight'][0,0]=.001;cp['current_calibration']['motion_projection.weight'][0,0]=.01;save(path,cp);updated,current,r=c.apply_spatial_motion_dn(path,parts,cal,motion)
 assert updated[4] is not parts[4] and parts[4].projection.weight[0,0]==0
 assert r['actor_qualified'] is False and r['vehicle_control_sent'] is False and not any(p.requires_grad for p in current.parameters())


@pytest.mark.parametrize('bad',['parent','motion_norm','dtype','motion_norm_bound','native_entry','gate','authority','hash'])
def test_reject_changed_contract_without_mutating_inputs(tmp_path,monkeypatch,bad):
 path,cp,parts,cal,motion=fixture(tmp_path,monkeypatch);before={k:v.clone() for k,v in parts[4].state_dict().items()};s=cp['current_calibration']
 if bad=='parent':s['parent.registration.projection.weight'][0,0]=.01
 if bad=='motion_norm':s['motion_scale'][0]=2
 if bad=='dtype':s['motion_projection.weight']=s['motion_projection.weight'].double()
 if bad=='motion_norm_bound':s['motion_projection.weight'][0,:2]=1.5
 if bad=='native_entry':cp['body_port']['projection.weight'][0,0]=.021
 if bad=='gate':s['parent.registration.gate'][0]=0
 if bad=='authority':cp['actor_qualified']=True
 save(path,cp)
 if bad=='hash':path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError):c.apply_spatial_motion_dn(path,parts,cal,motion)
 assert all(torch.equal(parts[4].state_dict()[k],v) for k,v in before.items())
