import copy
import pytest
import torch
from flybrain_drive.route_protected_visual_overlay import require_pooled_only_model


def states():
    original={'visual.weight':torch.zeros(2,588),'visual.bias':torch.zeros(2),'body.weight':torch.ones(2,2),'chemical':torch.sparse_coo_tensor(torch.tensor([[0,20000],[1,30000]]),torch.tensor([.5,.2]),(50000,50000))};candidate=copy.deepcopy(original);candidate['visual.weight'][0,12]=.0005
    return candidate,original


def test_pooled_sensory_columns_are_the_only_permitted_model_change():
    require_pooled_only_model(*states(),.001)


@pytest.mark.parametrize('problem',['bias','body','command','bound','edges'])
def test_route_overlay_protects_physical_graph_body_bias_and_nonvisual_columns(problem):
    candidate,parent=states()
    if problem=='bias':candidate['visual.bias'][0]=.001
    if problem=='body':candidate['body.weight'][0,0]=2
    if problem=='command':candidate['visual.weight'][0,0]=.001
    if problem=='bound':candidate['visual.weight'][0,12]=.002
    if problem=='edges':candidate['chemical']=torch.sparse_coo_tensor(parent['chemical']._indices(),parent['chemical']._values()+.1,parent['chemical'].shape)
    with pytest.raises(ValueError):require_pooled_only_model(candidate,parent,.001)
