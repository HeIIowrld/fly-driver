import copy
import numpy as np
import pytest
import torch
from torch import nn
from flybrain_drive.evidence import write_json
from flybrain_drive.functional_bend_objective import functional_bend_loss
from flybrain_drive.functional_motion_overlay import load_functional_motion_overlay,exact_model
from flybrain_drive.graph import sha256
from flybrain_drive.motion_spatial_residual_port import MotionSpatialResidualPort
from flybrain_drive.sac_unified_student import read
from flybrain_drive.synthetic_antialiased_planar_road_visual import AntialiasedSyntheticPlanarRoadVisual
from scripts.evaluate_controlled_motion_factorial import synthetic_body


def test_functional_loss_moves_opposite_bends_in_correct_hypothesized_order():
    right=torch.zeros(4,3,requires_grad=True);left=torch.zeros(4,3,requires_grad=True);anchor=torch.zeros(4,3)
    loss=functional_bend_loss(right,left,anchor,anchor,left_sign=-1,target_margin=.002,direction_weight=100.,anchor_weight=.5);loss.backward()
    assert (left.grad[:,0]>0).all() and (right.grad[:,0]<0).all()
    assert torch.count_nonzero(left.grad[:,1:])==0 and torch.count_nonzero(right.grad[:,1:])==0


def test_targets_are_detached_and_adequate_order_has_no_direction_loss():
    right=torch.tensor([[.003,0.,0.]],requires_grad=True);left=torch.tensor([[-.003,0.,0.]],requires_grad=True);ra=right.detach().clone().requires_grad_(True);la=left.detach().clone().requires_grad_(True)
    loss=functional_bend_loss(right,left,ra,la,left_sign=-1,target_margin=.002,direction_weight=100.,anchor_weight=.5);loss.backward()
    assert loss==0 and ra.grad is None and la.grad is None


def test_synthetic_temperature_and_speed_are_consistent_in_both_body_boundaries():
    reference=np.arange(19,dtype=np.float32);original=reference.copy();body,mask,native,valid=synthetic_body(6.,110.,reference)
    np.testing.assert_array_equal(reference,original);np.testing.assert_array_equal(body[12:16],[110]*4);np.testing.assert_array_equal(native[11:15],[110]*4);np.testing.assert_allclose(body[17:21],6/.29);assert body[16]==6 and not mask[2:4].any() and not mask[5:8].any() and valid.all()


def test_single_view_optimization_preserves_original_renderer_pixels():
    road=AntialiasedSyntheticPlanarRoadVisual(read('configs/ac-csp-exterior-front-v3-retina.json'),samples_per_axis=1);kwargs=dict(bend=1,forward_m=3.,lateral_m=-2.,yaw_rad=0.,texture_seed=781104);full=road.render(**kwargs)
    for j in range(4):
        view=copy.copy(road);view.rays=[road.rays[j]];view.origins=[road.origins[j]];np.testing.assert_array_equal(view.render(**kwargs)[0],full[j])


def overlay_fixture(tmp_path,problem=None):
    components=[nn.Linear(3,2) for _ in range(5)];motion=MotionSpatialResidualPort(torch.zeros(2048),torch.ones(2048),2);receipt={'motion_checkpoint_sha256':'fixture'};state=copy.deepcopy(motion.state_dict());state['projection.weight'][0,0]=.0005
    cp=dict(schema='flybrain.functional_motion_port.v1',parent_candidate=receipt,actor_qualified=False,optimizer_updates=18,maximum_weight_change=.001,motion_port=state,model=copy.deepcopy(components[0].state_dict()),**{key:copy.deepcopy(components[i].state_dict()) for key,i in [('readout',1),('adapter',2),('optical',3),('body_port',4)]})
    if problem=='model':cp['model']['weight'][0,0]+=1
    if problem=='normalization':state['reference'][0]+=1
    if problem=='bound':state['projection.weight'][0,0]=.002
    if problem=='qualified':cp['actor_qualified']=True
    path=tmp_path/'functional-motion.pt';torch.save(cp,path);write_json(tmp_path/'result.json',dict(status='completed_controlled_motion_function_fit',checkpoint_sha256=sha256(path),optimizer_updates=18));write_json(tmp_path/'manifest.json',dict(config=dict(parent_candidate=receipt,plan=dict(maximum_weight_change=.001))))
    return path,components,motion,receipt


def test_functional_overlay_is_readonly_research_only(tmp_path):
    args=overlay_fixture(tmp_path);receipt=load_functional_motion_overlay(*args);assert receipt['actor_qualified'] is False and receipt['vehicle_control_sent'] is False;assert args[2].projection.weight[0,0]==.0005


def test_sparse_frozen_model_identity_checks_edges_without_dense_materialization():
    value=torch.sparse_coo_tensor(torch.tensor([[0,20000],[1,30000]]),torch.tensor([.5,.2]),(50000,50000));assert exact_model({'edges':value},{'edges':value.clone()})
    changed=torch.sparse_coo_tensor(value._indices(),value._values()+.01,value.shape)
    assert not exact_model({'edges':value},{'edges':changed})


@pytest.mark.parametrize('problem',['model','normalization','bound','qualified'])
def test_functional_overlay_rejects_changed_parent_or_unqualified_contract(tmp_path,problem):
    args=overlay_fixture(tmp_path,problem)
    with pytest.raises(ValueError):load_functional_motion_overlay(*args)
    assert torch.count_nonzero(args[2].projection.weight)==0
