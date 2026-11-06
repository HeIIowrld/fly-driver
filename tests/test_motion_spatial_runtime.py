import numpy as np
import pytest
import torch
from torch import nn
from test_native_flight_decision_runtime import Model,Encoder,inputs
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.sac_unified_student import MuscleVehicleReadout
from flybrain_drive.motion_spatial_residual_port import MotionSpatialResidualPort
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.motion_spatial_runtime import MotionSpatialDecisionRuntime,MotionSpatialObservation


def make(clock=None):
    model=Model();kwargs={} if clock is None else dict(monotonic=clock)
    runtime=MotionSpatialDecisionRuntime(model,MuscleVehicleReadout(6),nn.Linear(320,1),RetinalResidualPort(torch.zeros(256),torch.ones(256),2),MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),2),Encoder(),motion_port=MotionSpatialResidualPort(torch.zeros(2048),torch.ones(2048),2),**kwargs)
    runtime.begin_episode('one');return runtime,model


def obs(kw):
    c=kw['body_context'];return MotionSpatialObservation('one',c.camera_receipt_time,c.render_start_lowers,np.zeros(512),np.ones(2048))


def test_motion_hook_keeps_continuous_state_and_discards_only_temporary_input():
    runtime,model=make();kw=inputs();out=runtime.decide(motion_observation=obs(kw),**kw)
    assert out['muscles'][0]==1 and runtime.motion_active is None
    kw=inputs(1.01,2);out=runtime.decide(motion_observation=obs(kw),**kw)
    assert out['muscles'][0]==2 and model.count==2
    runtime.close();assert not model.visual._forward_hooks


def test_total_deadline_includes_binding_outside_parent_runtime():
    runtime,model=make(iter([0.,.01,.015,.06]).__next__);kw=inputs()
    with pytest.raises(ValueError,match='Motion binding'):runtime.decide(motion_observation=obs(kw),**kw)
    assert runtime.poisoned and model.count==1 and runtime.last_rejection['computation']['includes_motion_binding']
    with pytest.raises(RuntimeError):runtime.decide(motion_observation=obs(kw),**kw)
    runtime.close()


def test_wrong_camera_rejected_before_neural_state_advance():
    runtime,model=make();kw=inputs();other=MotionSpatialObservation('one',.9,(.8,)*4,np.zeros(512),np.ones(2048))
    with pytest.raises(ValueError):runtime.decide(motion_observation=other,**kw)
    assert runtime.poisoned and model.count==0
    runtime.close()
