import pytest
import torch
from test_native_flight_body_ticker import make
from test_native_flight_decision_runtime import Model,Encoder
from flybrain_drive.motion_spatial_runtime import MotionSpatialDecisionRuntime
from flybrain_drive.motion_spatial_residual_port import MotionSpatialResidualPort
from flybrain_drive.motion_per_view_flyvis_stream import MotionVisualFlightObservation
from flybrain_drive.sac_unified_student import MuscleVehicleReadout
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort


def runtime():
    from torch import nn
    model=Model();model.previous_body_receipt=None
    r=MotionSpatialDecisionRuntime(model,MuscleVehicleReadout(6),nn.Linear(320,1),RetinalResidualPort(torch.zeros(256),torch.ones(256),2),MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),2),Encoder(),motion_port=MotionSpatialResidualPort(torch.zeros(2048),torch.ones(2048),2));r.begin_episode('one');return r


def test_motion_ticker_rejects_missing_motion_before_publication():
    ticker,_,rows,clock,visual,context=make();ticker.runtime=runtime()
    with pytest.raises(ValueError,match='typed motion'):ticker.publish(visual,context)
    assert ticker.version==0
    ticker.runtime.close()


def test_same_camera_motion_reaches_cpu_owner_decision_and_is_hash_bound():
    import numpy as np
    ticker,_,rows,clock,visual,context=make();ticker.runtime=runtime();obs=MotionVisualFlightObservation(**vars(visual),motion_features=np.zeros(2048),coarse_motion_reference=np.zeros(512))
    ticker.publish(obs,context);event=ticker.step_once()
    assert event['status']=='shadow_output' and event['motion_features_sha256']==obs.motion_features_sha256
    assert ticker.runtime.motion_active is None
    ticker.runtime.close()
