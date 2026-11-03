import numpy as np
import pytest
import torch
from types import SimpleNamespace
from flybrain_drive.graph import sha256
from flybrain_drive.interval_yaw_visual_boundary import IntervalYawVisualBoundary
from flybrain_drive.view_yaw_interval import ViewYawInterval


def test_per_view_mask_prevents_cross_view_imputation():
    digest=sha256('configs/ac-virtual-body-v1.json')
    boundary=IntervalYawVisualBoundary(1,body_contract_sha256=digest)
    with torch.no_grad():
        boundary.projection.weight[0,0]=1.
        boundary.projection.weight[0,64]=1.
    fast=torch.zeros((1,4,2,4,16));fast[0,0,0,1,0]=1.;fast[0,1,0,1,0]=2.
    traces=(fast.reshape(1,512),torch.zeros((1,512)),torch.zeros((1,512)))
    yaw=torch.tensor([[.2,.4,0.,0.]])
    front=boundary(traces,yaw,torch.tensor([[True,False,False,False]]))
    left=boundary(traces,yaw,torch.tensor([[False,True,False,False]]))
    missing=boundary(traces,yaw,torch.zeros((1,4),dtype=torch.bool))
    assert front.item()==pytest.approx(.5*np.tanh(.2))
    assert left.item()==pytest.approx(.5*np.tanh(.8))
    assert missing.item()==0.


def test_interval_requires_correct_view_bound_and_unqualified_source():
    digest=sha256('configs/ac-virtual-body-v1.json')
    boundary=IntervalYawVisualBoundary(1,body_contract_sha256=digest)
    ctx=SimpleNamespace(camera_receipt_time=1.08,render_start_lowers=(1.05,1.06,1.07,1.075))
    item=ViewYawInterval(1.,1.05,1.08,1.1,.2,.01,1,3,.99,1.06,digest)
    y,m=boundary.pack((item,None,None,None),body_context=ctx,decision_time=1.1)
    assert y[0,0].item()==pytest.approx(.2) and m.tolist()==[[True,False,False,False]]
    with pytest.raises(ValueError,match='provenance'):
        boundary.pack((None,item,None,None),body_context=ctx,decision_time=1.1)
    claimed=ViewYawInterval(1.,1.05,1.08,1.1,.2,.01,1,3,.99,1.06,digest,physical_time_verified=True)
    with pytest.raises(ValueError,match='provenance'):
        boundary.pack((claimed,None,None,None),body_context=ctx,decision_time=1.1)
