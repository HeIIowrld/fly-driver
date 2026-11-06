"""Synthetic checkpoint contract checks; no measured graph or simulator."""
import json
import pytest
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
import flybrain_drive.fixed_full_routing_candidate as c


def fixture(tmp_path,monkeypatch):
 paths={}
 for key in ['parent_checkpoint','parent_port_checkpoint','fixed_decoder']:
  p=tmp_path/(key+'.pt');p.write_bytes(key.encode());paths[key]=str(p);paths[key+'_sha256']=sha256(p)
 cal=VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.]))
 body=MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),3);components=tuple([torch.nn.Identity() for _ in range(4)]+[body,object()]);monkeypatch.setattr(c,'apply_visual_native_ports',lambda path,parts,base:(parts,base,{}));current=c.FixedFullRoutingCurrent(cal,1.)
 plan=dict(paths,initial_gate=1.,minimum_gate=1.,visual_total_norm=2.,bias_total_norm=2.,native_total_norm=.05,native_entry_delta=.02)
 cp=dict(schema='flybrain.fixed_full_routing_ports.experimental.v1',arm='visual_native',current_calibration=current.state_dict(),body_port=body.state_dict(),actor_qualified=False,**{k:plan[k] for k in ['fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','initial_gate','minimum_gate']})
 path=tmp_path/'ports.pt';(tmp_path/'manifest.json').write_text(json.dumps({'config':dict(plan=plan,arm='visual_native')}));return path,cp,components,cal


def save(path,cp):
 torch.save(cp,path);(path.parent/'result.json').write_text(json.dumps(dict(status='completed_fixed_full_routing_fit',checkpoint_sha256=sha256(path))))


def test_replay_overlay_uses_private_body_and_grants_no_authority(tmp_path,monkeypatch):
 path,cp,parts,cal=fixture(tmp_path,monkeypatch);cp['body_port']['projection.weight']=cp['body_port']['projection.weight'].clone();cp['body_port']['projection.weight'][0,0]=.001;save(path,cp);updated,current,r=c.apply_fixed_full_routing(path,parts,cal)
 assert updated[4] is not parts[4] and parts[4].projection.weight[0,0]==0
 assert r['actor_qualified'] is False and r['vehicle_control_sent'] is False
 assert not any(p.requires_grad for p in current.parameters())


@pytest.mark.parametrize('bad',['base','scale','bias','gate','native','authority','hash','almost_gate'])
def test_reject_contract_changes_without_mutating_inputs(tmp_path,monkeypatch,bad):
 path,cp,parts,cal=fixture(tmp_path,monkeypatch);before={k:v.clone() for k,v in parts[4].state_dict().items()};s=cp['current_calibration']
 if bad=='base':s['base.projection.weight'][0,0]=.01
 if bad=='scale':s['registration.scale'][0]=2
 if bad=='bias':s['registration.projection.bias'][0]=2.1
 if bad=='almost_gate':s['registration.gate'][0]=1.-torch.finfo(torch.float32).eps/2
 if bad=='gate':s['registration.gate'][0]=0
 if bad=='native':cp['body_port']['projection.weight']=cp['body_port']['projection.weight'].clone();cp['body_port']['projection.weight'][0,0]=.021
 if bad=='authority':cp['actor_qualified']=True
 save(path,cp)
 if bad=='hash':path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError):c.apply_fixed_full_routing(path,parts,cal)
 assert all(torch.equal(parts[4].state_dict()[k],v) for k,v in before.items())
