from dataclasses import replace
import pytest
import torch
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from flybrain_drive.unified_multirate_shadow import UnifiedMultirateShadow
from test_unified_flight_sensorimotor import context, make_model


class Clock:
    def __init__(self, times):self.times=iter(times)
    def __call__(self):return next(self.times)


def test_original_scene_is_copied_held_and_priming_suppressed():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    shadow=UnifiedMultirateShadow(UnifiedHostGuard(model,clock=Clock([1.,1.001,1.03,1.031])),goals=torch.tensor([[0.,1.]]))
    original=torch.zeros(1,576);frame=context(1.)
    assert shadow.update(when=1.,body=frame.decision_body,new_scene=original,new_context=frame,published_at=1.) is None
    original.fill_(1.)
    assert torch.count_nonzero(shadow.scene)==0
    out=shadow.update(when=1.03,body=context(1.03).decision_body)
    assert out is not None and out.actor_qualified is False


def test_reject_future_publication_and_incomplete_scene():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    shadow=UnifiedMultirateShadow(UnifiedHostGuard(model,clock=Clock([])),goals=torch.tensor([[0.,1.]]));frame=context(1.)
    with pytest.raises(ValueError,match='publication'):
        shadow.update(when=1.,body=frame.decision_body,new_scene=torch.zeros(1,576),new_context=frame,published_at=1.01)
    with pytest.raises(ValueError,match='together'):
        shadow.update(when=1.,body=frame.decision_body,new_scene=torch.zeros(1,576))
    assert model.state is None


def test_output_timeout_latches_and_requires_restart():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    shadow=UnifiedMultirateShadow(UnifiedHostGuard(model,clock=Clock([1.,1.001,1.03,1.08,1.1,1.101])),goals=torch.tensor([[0.,1.]]));frame=context(1.)
    shadow.update(when=1.,body=frame.decision_body,new_scene=torch.zeros(1,576),new_context=frame,published_at=1.)
    with pytest.raises(ValueError,match='stale at output'):shadow.update(when=1.03,body=context(1.03).decision_body)
    with pytest.raises(RuntimeError,match='restart'):shadow.update(when=1.04,body=context(1.04).decision_body)
    shadow.reset();frame=context(1.1)
    assert shadow.update(when=1.1,body=frame.decision_body,new_scene=torch.zeros(1,576),new_context=frame,published_at=1.1) is None
