from dataclasses import replace
import pytest
from flybrain_drive.current_flight_control_contract import FlightDecisionEnvelope,control_request

IDENTITY=dict(run_id='owned',episode_id='one',generation='g',checkpoint_sha256='a'*64)
def envelope():
    return FlightDecisionEnvelope(**IDENTITY,decision_id=1,publication_id=1,capture_id=4,
        body_packet=12,body_first_receipt=1.,view_render_lowers=(.99,)*4,
        view_render_uppers=(.992,)*4,view_body_first_copy_starts=(.98,)*4,
        camera_received=.995,published_at=.999,decision_started=1.001,
        inference_finished=1.01,prediction=(-.5,.2,.1))

def test_original_teacher_scale_is_inverted_exactly_once_and_pedals_are_independent():
    r=control_request([-.5,.2,.1])
    assert r['requested_control']==dict(steering=-.3,throttle=.2,brake=.1)
    assert not r['clipping_applied'] and not r['pedal_override']
    assert control_request([1,0,0])['requested_control']==dict(steering=.6,throttle=0.,brake=0.)
    assert control_request([0,0,1])['requested_control']==dict(steering=0.,throttle=0.,brake=1.)

@pytest.mark.parametrize('value',[[0,0],[float('nan'),0,0],[0,-.1,0],[0,0,1.1],[1.1,0,0]])
def test_invalid_outputs_are_rejected_without_clipping(value):
    with pytest.raises(ValueError):control_request(value)

def test_consumption_uses_original_source_age_not_inference_or_rpc_receipt():
    v=envelope();assert v.validate(now=1.02,**IDENTITY)==1.05
    with pytest.raises(ValueError,match='freshness'):v.validate(now=1.051,**IDENTITY)
    with pytest.raises(ValueError,match='freshness'):replace(v,view_render_lowers=(.8,)*4).validate(now=1.02,**IDENTITY)

@pytest.mark.parametrize('change',[
    dict(generation='other'),dict(checkpoint_sha256='b'*64),dict(decision_id=0),
    dict(body_first_receipt=1.03),dict(inference_finished=1.03),
    dict(view_render_uppers=(1.,)*4,view_body_first_copy_starts=(.97,)*4),
    dict(camera_received=float('nan')),dict(view_render_lowers=(.99,)*3)])
def test_invalid_identity_source_and_future_are_rejected(change):
    with pytest.raises(ValueError):replace(envelope(),**change).validate(now=1.02,**IDENTITY)

def test_same_camera_can_support_new_body_decision_but_same_decision_cannot_be_reapplied():
    v=replace(envelope(),decision_id=2)
    v.validate(now=1.02,**IDENTITY,after_decision=1,after_publication=1,after_capture=4)
    with pytest.raises(ValueError):v.validate(now=1.02,**IDENTITY,after_decision=2)


def test_feature_selection_margin_does_not_extend_or_restamp_the_source():
    import copy
    from flybrain_drive.ac_csp_vision import candidate_at,NoFreshCSPFrame
    from flybrain_drive.current_flight_control_contract import FEATURE_SELECTION_MAX_VIEW_AGE
    row=dict(capture_id=1,received=.99,views=[dict(render_started_lower=.97) for _ in range(4)])
    before=copy.deepcopy(row)
    assert candidate_at([row],1.,.15) is row
    with pytest.raises(NoFreshCSPFrame):candidate_at([row],1.,FEATURE_SELECTION_MAX_VIEW_AGE)
    assert row==before
