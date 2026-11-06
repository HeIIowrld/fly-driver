import copy,json
import pytest
import torch
from torch import nn
from flybrain_drive.graph import sha256
from flybrain_drive.visual_descending_port import VisualDescendingPort
from flybrain_drive.visual_flight_speed_intent import VisualFlightSpeedIntent
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.turn_speed_candidate import load_turn_speed

@pytest.fixture
def bundle(tmp_path):
 components=[nn.Linear(2,2),nn.Linear(2,3),VisualDescendingPort(torch.zeros(320),torch.ones(320),torch.tensor([-1.,1.,-1.,1.])),nn.Linear(2,2),MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),2),None];motion=nn.Linear(2,2);receipt={'parent':'fixed'};speed=VisualFlightSpeedIntent(components[2].reference,components[2].scale,components[4].reference,components[4].scale)
 cp=dict(schema='flybrain.turn_speed_intent.v1',actor_qualified=False,parent_candidate=receipt,optimizer_updates=147,speed_intent=copy.deepcopy(speed.state_dict()),speed_active=True,speed_limits=dict(max_delta_m_s=10.,maximum_log_gain=.5))
 for key,index in [('model',0),('readout',1),('adapter',2),('optical',3),('body_port',4)]:cp[key]=copy.deepcopy(components[index].state_dict())
 cp['motion_port']=copy.deepcopy(motion.state_dict());cp['speed_intent']['visual_projection.weight'].fill_(.001)
 def write(candidate):
  path=tmp_path/'turn-speed.pt';torch.save(candidate,path);(tmp_path/'result.json').write_text(json.dumps(dict(status='completed_turn_speed_intent_fit',arm='turn_speed',checkpoint_sha256=sha256(path),optimizer_updates=147)));(tmp_path/'manifest.json').write_text(json.dumps(dict(config=dict(parent_candidate=receipt,plan=dict(maximum_weight_delta=.025,maximum_speed_weight_delta=.025,maximum_speed_delta_m_s=10.,maximum_body_log_gain=.5)))));return path
 return components,motion,receipt,cp,write

def test_allowed_speed_weights_preserve_graph_and_native_body(bundle):
 c,m,r,cp,write=bundle;original=copy.deepcopy(c[0].state_dict());receipt,speed,active=load_turn_speed(write(cp),c,m,r)
 assert active and not receipt['actor_qualified'] and not receipt['vehicle_control_sent']
 assert all(torch.equal(v,c[0].state_dict()[k]) for k,v in original.items())
 torch.testing.assert_close(speed.visual_projection.weight,torch.full((1,320),.001))

@pytest.mark.parametrize('fault',['turn_bound','speed_bound','gain_bound','nan','normalization','body','readout','mode','limit','parent'])
def test_rejects_unregistered_changes_before_applying(bundle,fault):
 c,m,r,cp,write=bundle
 if fault=='turn_bound':cp['adapter']['projection.weight'].fill_(.026)
 elif fault=='speed_bound':cp['speed_intent']['visual_projection.weight'].fill_(.026)
 elif fault=='gain_bound':cp['speed_intent']['body_gain.weight'].fill_(.026)
 elif fault=='nan':cp['speed_intent']['body_gain.weight'][0,0]=float('nan')
 elif fault=='normalization':cp['speed_intent']['native_scale'][0]=2
 elif fault=='body':cp['body_port']['projection.weight']+=1
 elif fault=='readout':cp['readout']['weight']+=1
 elif fault=='mode':cp['speed_active']=False
 elif fault=='limit':cp['speed_limits']['max_delta_m_s']=20
 elif fault=='parent':cp['parent_candidate']={'other':True}
 with pytest.raises(ValueError):load_turn_speed(write(cp),c,m,r)
 assert torch.count_nonzero(c[2].projection.weight)==0
