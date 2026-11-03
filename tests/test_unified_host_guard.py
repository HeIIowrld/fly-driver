from dataclasses import replace

import pytest
import torch

from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from test_unified_flight_sensorimotor import context, make_model


class Clock:
    def __init__(self, values):self.values=iter(values)
    def __call__(self):return next(self.values)


def call(guard, now):
    return guard.step(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),
        scene_features=torch.zeros(1,576),scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=context(now))


def test_priming_suppressed_and_fresh_output_is_unqualified():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    guard=UnifiedHostGuard(model,clock=Clock([1.,1.001,1.03,1.031]))
    assert call(guard,1.) is None
    result=call(guard,1.03)
    assert result.completed_time == 1.031 and result.actor_qualified is False
    assert result.physical_synchronization_verified is False


def test_stale_after_computation_latches_failure_and_restart_is_explicit():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    guard=UnifiedHostGuard(model,clock=Clock([1.,1.001,1.03,1.08,1.1,1.101]))
    assert call(guard,1.) is None
    with pytest.raises(ValueError,match='stale at output'):call(guard,1.03)
    assert guard.failed and model.previous_time==1.03
    with pytest.raises(RuntimeError,match='explicit restart'):call(guard,1.04)
    guard.reset();assert model.previous_time is None
    assert call(guard,1.1) is None


def test_reject_missing_limits_and_backwards_clock():
    with pytest.raises(ValueError):UnifiedHostGuard(make_model())
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    guard=UnifiedHostGuard(model,clock=Clock([.9]))
    with pytest.raises(ValueError,match='future'):call(guard,1.)
    assert guard.failed and model.previous_time is None


def test_view_can_become_stale_during_priming_and_fails_closed():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    guard=UnifiedHostGuard(model,clock=Clock([1.,1.005]))
    ctx=replace(context(1.),render_start_lowers=(.852,)*4,camera_receipt_time=.97)
    with pytest.raises(ValueError,match='View render bound stale at output'):
        guard.step(decision_time=1.,racing_goals=torch.tensor([[0.,1.]]),
            scene_features=torch.zeros(1,576),scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=ctx)
    assert guard.failed and model.previous_time == 1.
