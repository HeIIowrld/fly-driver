"""Synthetic loader contracts only; no claim of a measured graph fixture."""
import json
from types import SimpleNamespace
import pytest
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import MuscleVehicleReadout
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.joint_dn_decoder_candidate import load_joint_dn_decoder

def fixture(tmp_path):
 parent={'synthetic_test_identity':1};head=MuscleVehicleReadout(6);dn=SimpleNamespace(reference=torch.zeros(320),scale=torch.ones(320),orientations=torch.tensor([1.,-1.,1.,-1.]),maximum=.2);cal=VisualDNCurrentCalibration(dn.reference,dn.scale,dn.orientations,rank=4);hs={k:v.clone() for k,v in head.state_dict().items()};hs['projection.weight'][1,4:]=.1;hs['projection.weight'][2,4:]=-.1;candidate=tmp_path/'parent.json';candidate.write_text('{}');plan=dict(candidate_plan=str(candidate),candidate_plan_sha256=sha256(candidate),minimum_power_weight=.0001,maximum_current_weight=.05,maximum_logit_shift=2.);state=dict(schema='flybrain.joint_dn_decoder.experimental.v1',readout=hs,current_calibration=cal.state_dict(),rank=4,parent_candidate=parent,actor_qualified=False);path=tmp_path/'joint.pt';(tmp_path/'manifest.json').write_text(json.dumps({'config':dict(plan=plan,parent_candidate=parent,rank=4)}));components=(torch.nn.Identity(),head,dn,torch.nn.Identity(),torch.nn.Identity(),object());return path,state,components,parent

def save(path,state):
 torch.save(state,path);(path.parent/'result.json').write_text(json.dumps(dict(status='completed_joint_dn_decoder_fit',checkpoint_sha256=sha256(path))))

def test_valid_overlay_is_frozen_and_unqualified(tmp_path):
 path,state,components,parent=fixture(tmp_path);save(path,state);result,cal,receipt=load_joint_dn_decoder(path,components,parent);assert receipt['actor_qualified'] is False and receipt['vehicle_control_sent'] is False;assert not any(p.requires_grad for p in [*result[1].parameters(),*cal.parameters()])

@pytest.mark.parametrize('change',['reference','dtype','power','rank','authority','hash'])
def test_reject_invalid_overlay(tmp_path,change):
 path,state,components,parent=fixture(tmp_path)
 if change=='reference':state['readout']['training_reference'][0]=1
 if change=='dtype':state['readout']['projection.weight']=state['readout']['projection.weight'].double()
 if change=='power':state['readout']['projection.weight'][1,4]=-.1
 if change=='rank':state['rank']=1
 if change=='authority':state['actor_qualified']=True
 save(path,state)
 if change=='hash':path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError):load_joint_dn_decoder(path,components,parent)
