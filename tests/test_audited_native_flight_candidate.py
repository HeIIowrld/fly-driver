import copy
import pytest
import torch
from flybrain_drive.audited_native_flight_candidate import require_preserved_native_components


def states():
    native={'body_port':{'projection.weight':torch.ones(2,3)}}
    parent={'readout':{'training_reference':torch.zeros(6),'training_scale':torch.ones(6),
        'projection.weight':torch.ones(3,6),'projection.bias':torch.zeros(3)}}
    candidate=copy.deepcopy(native);candidate.update(copy.deepcopy(parent))
    candidate['readout']['projection.weight'][0]*=2
    return candidate,native,parent


def test_steering_only_change_preserves_frozen_body_and_pedals():
    require_preserved_native_components(*states())


def test_equal_values_with_changed_sensor_dtype_are_not_exact_preservation():
    candidate,native,parent=states();candidate['body_port']['projection.weight']=candidate['body_port']['projection.weight'].double()
    with pytest.raises(ValueError):require_preserved_native_components(candidate,native,parent)


@pytest.mark.parametrize('component,key,index',[
    ('body_port','projection.weight',(0,0)),('readout','projection.weight',(1,0)),
    ('readout','projection.bias',2),('readout','training_reference',0),('readout','training_scale',0)])
def test_unrelated_changes_cannot_be_loaded_as_protected_candidate(component,key,index):
    candidate,native,parent=states();candidate[component][key][index]+=1
    with pytest.raises(ValueError):require_preserved_native_components(candidate,native,parent)
