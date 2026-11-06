"""Synthetic artifact contracts; no measured circuit fixture is asserted."""
import json
import pytest
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.muscle_interaction_vehicle_readout import MuscleInteractionVehicleReadout
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head

def fixture(tmp_path):
 state=dict(training_reference=torch.zeros(6),training_scale=torch.ones(6),**{'projection.weight':torch.zeros(3,6),'projection.bias':torch.zeros(3)});state['projection.weight'][1,4:]=.1;state['projection.weight'][2,4:]=-.1;parent=tmp_path/'parent.pt';torch.save({'readout':state},parent);head=MuscleInteractionVehicleReadout(state,degree=2,quadratic_reference=torch.zeros(21,dtype=torch.float64),quadratic_scale=torch.ones(21,dtype=torch.float64));path=tmp_path/'decoder.pt';plan=dict(parent_checkpoint=str(parent),parent_checkpoint_sha256=sha256(parent),coefficient_bound=10.);(tmp_path/'manifest.json').write_text(json.dumps({'config':{'protocol':plan}}));cp=dict(schema='flybrain.muscle_interaction_decoder.experimental.v1',degree=2,readout=head.state_dict(),parent_checkpoint_sha256=sha256(parent),actor_qualified=False);return path,cp
def save(path,cp):
 torch.save(cp,path);(path.parent/'result.json').write_text(json.dumps(dict(status='completed_multicorner_constrained_decoder_fit',variant='quadratic',checkpoint_sha256=sha256(path))))
def test_loader_freezes_without_authority(tmp_path):
 path,cp=fixture(tmp_path);save(path,cp);head,receipt=load_muscle_interaction_head(path);assert not any(p.requires_grad for p in head.parameters()) and receipt['actor_qualified'] is False
@pytest.mark.parametrize('change',['base','dtype','power','authority','bound','hash'])
def test_reject_changed_contract(tmp_path,change):
 path,cp=fixture(tmp_path)
 if change=='base':cp['readout']['base.training_reference'][0]=1
 if change=='dtype':cp['readout']['steering.weight']=cp['readout']['steering.weight'].float()
 if change=='power':cp['readout']['base.projection.weight'][1,4]=-.1
 if change=='authority':cp['actor_qualified']=True
 if change=='bound':cp['readout']['steering.weight'][0,0]=11
 save(path,cp)
 if change=='hash':path.write_bytes(path.read_bytes()+b'changed')
 with pytest.raises(ValueError):load_muscle_interaction_head(path)
