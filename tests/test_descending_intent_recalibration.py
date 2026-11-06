"""Overlay permits only bounded original intent weights; rejects other changes."""
import copy,json
import pytest
import torch
from torch import nn
from flybrain_drive.graph import sha256
from flybrain_drive.visual_descending_port import VisualDescendingPort
from flybrain_drive.descending_intent_recalibration import load_descending_intent

@pytest.fixture
def bundle(tmp_path):
 components=[nn.Linear(2,2),nn.Linear(2,3),VisualDescendingPort(torch.zeros(320),torch.ones(320),torch.tensor([-1.,1.,-1.,1.])),nn.Linear(2,2),nn.Linear(2,2),None];motion=nn.Linear(2,2);receipt={'parent':'fixed'}
 cp=dict(schema='flybrain.descending_intent.v1',actor_qualified=False,parent_candidate=receipt,optimizer_updates=147)
 for key,index in [('model',0),('readout',1),('adapter',2),('optical',3),('body_port',4)]:cp[key]=copy.deepcopy(components[index].state_dict())
 cp['motion_port']=copy.deepcopy(motion.state_dict());cp['adapter']['projection.weight']+=.01
 def write(candidate):
  path=tmp_path/'descending-intent.pt';torch.save(candidate,path)
  (tmp_path/'result.json').write_text(json.dumps(dict(status='completed_descending_intent_fit',checkpoint_sha256=sha256(path),optimizer_updates=147)))
  (tmp_path/'manifest.json').write_text(json.dumps(dict(config=dict(parent_candidate=receipt,plan=dict(maximum_weight_delta=.025)))))
  return path
 return components,motion,receipt,cp,write

def test_allowed_change_still_uses_original_module(bundle):
 components,motion,receipt,cp,write=bundle;original=copy.deepcopy(components[0].state_dict());result=load_descending_intent(write(cp),components,motion,receipt)
 assert result['actor_qualified'] is False and result['vehicle_control_sent'] is False
 torch.testing.assert_close(components[2].projection.weight,torch.full((1,320),.01))
 assert all(torch.equal(v,components[0].state_dict()[k]) for k,v in original.items())
 assert not any(p.requires_grad for p in components[2].parameters())

@pytest.mark.parametrize('fault',['bound','nan','shape','body','motion','normalization','readout','parent','qualified'])
def test_rejects_invalid_or_unrelated_changes_before_application(bundle,fault):
 components,motion,receipt,cp,write=bundle
 if fault=='bound':cp['adapter']['projection.weight'].fill_(.026)
 elif fault=='nan':cp['adapter']['projection.weight'][0,0]=float('nan')
 elif fault=='shape':cp['adapter']['projection.weight']=torch.zeros(1,321)
 elif fault=='body':cp['body_port']['weight']+=1
 elif fault=='motion':cp['motion_port']['weight']+=1
 elif fault=='normalization':cp['adapter']['scale'][0]=2
 elif fault=='readout':cp['readout']['weight']+=1
 elif fault=='parent':cp['parent_candidate']={'different':True}
 elif fault=='qualified':cp['actor_qualified']=True
 with pytest.raises(ValueError):load_descending_intent(write(cp),components,motion,receipt)
 assert torch.count_nonzero(components[2].projection.weight)==0
