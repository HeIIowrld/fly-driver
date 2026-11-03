from dataclasses import replace
import pytest
import torch

from flybrain_drive.temporal_body_context import HostFreshnessLimits, per_view_temporal_body_context
from test_unified_flight_sensorimotor import context, make_model


def call(model, now, ctx):
    return model.sensory_decision(decision_time=now, racing_goals=torch.tensor([[0.,1.]]),
        scene_features=torch.zeros(1,576),scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=ctx)


def test_freshness_limits_reject_bad_configuration():
    for b,v in ((0., .15),(.051,.15),(.05,.151),(True,.1)):
        with pytest.raises(ValueError): HostFreshnessLimits(b,v)


def test_stale_current_or_view_does_not_advance_state():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.05,.15)
    call(model,1.,context(1.));before=model.state.clone()
    fresh=context(1.03);bad_body=replace(fresh.decision_body,original_received_monotonic_seconds=.95)
    stale=per_view_temporal_body_context((None,)*4,bad_body,render_start_lowers=(1.01,)*4,camera_receipt_time=1.02,decision_time=1.03)
    with pytest.raises(ValueError,match='Current body receipt stale'):call(model,1.03,stale)
    assert torch.equal(before,model.state) and model.previous_time==1.
    stale=per_view_temporal_body_context((None,)*4,fresh.decision_body,render_start_lowers=(.8,)*4,camera_receipt_time=.9,decision_time=1.03)
    with pytest.raises(ValueError,match='View render bound stale'):call(model,1.03,stale)
    assert torch.equal(before,model.state) and model.previous_time==1.


def test_stale_held_body_rejected_and_restart_clears_it():
    model=make_model();model.freshness_limits=HostFreshnessLimits(.02,.15)
    call(model,1.,context(1.));before=model.state.clone()
    with pytest.raises(ValueError,match='Held body receipt stale'):call(model,1.04,context(1.04))
    assert torch.equal(before,model.state) and model.previous_time==1.
    model.reset_episode();assert model.previous_body_receipt is None
    call(model,1.04,context(1.04))
