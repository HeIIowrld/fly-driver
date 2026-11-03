import numpy as np
import pytest
import torch
from flybrain_drive.process_unified_multirate_shadow import ProcessUnifiedMultirateShadow
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from test_unified_flight_sensorimotor import context,make_model

class Clock:
 def __init__(self,values):self.values=iter(values)
 def __call__(self):return next(self.values)

class Worker:
 def __init__(self,model):self.model=model;self.resets=0
 def step(self,*,when,scene,context):
  out,muscle=self.model.sensory_decision(decision_time=when,racing_goals=torch.tensor([[0.,1.]]),scene_features=torch.from_numpy(scene),scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=context)
  return {'intent':out.numpy(),'muscles':muscle.numpy()}
 def reset(self):self.model.reset_episode();self.resets+=1

def test_process_shadow_preserves_priming_state_and_reset():
 torch.manual_seed(9);model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15);other=make_model();other.load_state_dict(model.state_dict());other.freshness_limits=HostFreshnessLimits(.05,.15);worker=Worker(other)
 shadow=ProcessUnifiedMultirateShadow(UnifiedHostGuard(model),worker,goals=torch.tensor([[0.,1.]]),clock=Clock([1.,1.001,1.03,1.031]))
 frame=context(1.);scene=torch.zeros(1,576)
 assert shadow.update(when=1.,body=frame.decision_body,new_scene=scene,new_context=frame,published_at=1.) is None
 out=shadow.update(when=1.03,body=context(1.03).decision_body)
 assert out is not None and out.actor_qualified is False
 assert model.previous_time==other.previous_time==1.03
 shadow.reset();assert model.previous_time is None and other.previous_time is None and worker.resets==1

def test_process_shadow_latches_stale_response():
 model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15);other=make_model();other.load_state_dict(model.state_dict());other.freshness_limits=HostFreshnessLimits(.05,.15)
 shadow=ProcessUnifiedMultirateShadow(UnifiedHostGuard(model),Worker(other),goals=torch.tensor([[0.,1.]]),clock=Clock([1.,1.001,1.03,1.08]));frame=context(1.)
 shadow.update(when=1.,body=frame.decision_body,new_scene=torch.zeros(1,576),new_context=frame,published_at=1.)
 with pytest.raises(ValueError,match='stale at output'):shadow.update(when=1.03,body=context(1.03).decision_body)
 with pytest.raises(RuntimeError,match='restart'):shadow.update(when=1.04,body=context(1.04).decision_body)
