import copy
import pytest
import torch
from torch import nn
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.audited_native_visual_candidate import require_visual_only_overlay


def states():
    head=nn.Linear(6,3);dn=nn.Linear(320,4);body=nn.Linear(19,12)
    optical=RetinalResidualPort(torch.zeros(256),torch.ones(256),4)
    candidate=dict(readout=copy.deepcopy(head.state_dict()),adapter=copy.deepcopy(dn.state_dict()),
        body_port=copy.deepcopy(body.state_dict()),optical=copy.deepcopy(optical.state_dict()))
    candidate['optical']['projection.weight'][0,0]=.01
    return candidate,head,dn,body,optical


def test_overlay_changes_only_inherited_visual_projection():
    require_visual_only_overlay(*states(),.02)


@pytest.mark.parametrize('key,field',[('readout','weight'),('adapter','weight'),('body_port','bias'),('optical','reference'),('optical','scale')])
def test_body_dn_command_or_normalization_change_is_rejected(key,field):
    values=states();values[0][key][field].view(-1)[0]+=1
    with pytest.raises(ValueError):require_visual_only_overlay(*values,.02)


@pytest.mark.parametrize('problem',['dtype','nan','bound'])
def test_visual_value_shape_contract_and_bound_are_not_bypassed(problem):
    values=states();cp=values[0]
    if problem=='dtype':cp['optical']['projection.weight']=cp['optical']['projection.weight'].double()
    if problem=='nan':cp['optical']['projection.weight'][0,0]=float('nan')
    if problem=='bound':cp['optical']['projection.weight'][0,0]=.021
    with pytest.raises(ValueError):require_visual_only_overlay(*values,.02)
