"""Synthetic port loader contracts; fixture is not a biological graph."""
import json
import pytest
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
import flybrain_drive.visual_native_port_candidate as candidate

def fixture(tmp_path,monkeypatch):
 parent=tmp_path/'parent.pt';parent.write_bytes(b'synthetic parent');decoder=tmp_path/'decoder.pt';decoder.write_bytes(b'synthetic fixed decoder');cal=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]),rank=4);body=MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),3);components=(torch.nn.Identity(),torch.nn.Identity(),torch.nn.Identity(),torch.nn.Identity(),body,object());monkeypatch.setattr(candidate,'load_muscle_interaction_head',lambda path:(torch.nn.Identity(),{'actor_qualified':False}));plan=dict(fixed_decoder=str(decoder),fixed_decoder_sha256=sha256(decoder),parent_checkpoint=str(parent),parent_checkpoint_sha256=sha256(parent),visual_absolute_bound=.08,visual_total_norm=.1,native_total_norm=.01,native_entry_delta=.002);cp=dict(schema='flybrain.visual_native_ports.experimental.v1',arm='visual_native',current_calibration={k:v.clone() for k,v in cal.state_dict().items()},body_port={k:v.clone() for k,v in body.state_dict().items()},fixed_decoder=str(decoder),fixed_decoder_sha256=sha256(decoder),parent_checkpoint_sha256=sha256(parent),actor_qualified=False);path=tmp_path/'ports.pt';(tmp_path/'manifest.json').write_text(json.dumps({'config':dict(plan=plan,arm='visual_native')}));return path,cp,components,cal
def save(path,cp):
 torch.save(cp,path);(path.parent/'result.json').write_text(json.dumps(dict(status='completed_visual_native_port_fit',checkpoint_sha256=sha256(path))))
def test_valid_port_overlay_has_no_authority(tmp_path,monkeypatch):
 path,cp,components,cal=fixture(tmp_path,monkeypatch);cp['body_port']['projection.weight'][0,0]=.0001;save(path,cp);updated,cal,receipt=candidate.apply_visual_native_ports(path,components,cal);assert receipt['actor_qualified'] is False and receipt['vehicle_control_sent'] is False and not any(v.requires_grad for v in updated[4].parameters())
@pytest.mark.parametrize('change',['normalization','dtype','native_entry','visual_bound','authority','hash'])
def test_reject_changed_port_contract(tmp_path,monkeypatch,change):
 path,cp,components,cal=fixture(tmp_path,monkeypatch)
 if change=='normalization':cp['body_port']['reference'][0]=1
 if change=='dtype':cp['body_port']['projection.weight']=cp['body_port']['projection.weight'].double()
 if change=='native_entry':cp['body_port']['projection.weight'][0,0]=.003
 if change=='visual_bound':cp['current_calibration']['projection.weight'][0,0]=.09
 if change=='authority':cp['actor_qualified']=True
 save(path,cp)
 if change=='hash':path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError):candidate.apply_visual_native_ports(path,components,cal)
