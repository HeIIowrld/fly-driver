from dataclasses import replace
import pytest
from test_current_flight_control_contract import IDENTITY,envelope
from flybrain_drive.current_flight_actuation import FlightActuationLease

class Session:
    def __init__(self):self.requests=[];self.owned=True
    def submit(self,*args,**kw):
        value=args if args else tuple(kw[k] for k in ('steering','throttle','brake'))
        self.requests.append(value);return dict(driver_accepted=True,game_applied=False)
    def close(self):
        try:return self.submit(0.,0.,1.)
        finally:self.owned=False

def make():
    clock=[1.02];session=Session()
    lease=FlightActuationLease(session,identity=IDENTITY,clock=lambda:clock[0])
    return lease,session,clock

def test_preparation_has_no_student_controls_and_does_not_allow_early_arm():
    lease,session,clock=make()
    with pytest.raises(ValueError):lease.arm(prepared_since=.9,valid_camera_groups=3)
    assert session.requests==[]
    with pytest.raises(ValueError):lease.arm(prepared_since=.4,valid_camera_groups=2)
    lease.arm(prepared_since=.4,valid_camera_groups=3)
    assert session.requests==[]


def test_caller_stall_expires_original_lease_to_brake_and_remains_terminal():
    lease,session,clock=make();lease.arm(prepared_since=.4,valid_camera_groups=3)
    lease.apply(envelope());assert session.requests==[(-.3,.2,.1)]
    clock[0]=1.051;lease.poll()
    assert session.requests[-1]==(0.,0.,1.) and lease.stop_reason=='source_lease_expired'
    with pytest.raises(RuntimeError):lease.apply(replace(envelope(),decision_id=2))
    lease.release(stationary_feedback_verified=True)
    assert not session.owned and lease.state=='released'


def test_duplicate_output_stops_instead_of_refreshing_lease():
    lease,session,clock=make();lease.arm(prepared_since=.4,valid_camera_groups=3)
    lease.apply(envelope())
    with pytest.raises(ValueError):lease.apply(envelope())
    assert session.requests==[(-.3,.2,.1),(0.,0.,1.)]


def test_unknown_or_eof_stop_cannot_rearm():
    lease,session,clock=make();lease.arm(prepared_since=.4,valid_camera_groups=3)
    lease.stop('EOF unknown');lease.stop('another_reason')
    assert len(session.requests)==1 and lease.stop_reason=='EOF unknown'
    with pytest.raises(ValueError):lease.arm(prepared_since=.4,valid_camera_groups=3)


def test_sdk_latency_crossing_deadline_stops_after_retaining_request():
    lease,session,clock=make();lease.arm(prepared_since=.4,valid_camera_groups=3)
    submit=session.submit
    def late(*a,**kw):
        value=submit(*a,**kw);clock[0]+=.04;return value
    session.submit=late
    with pytest.raises(TimeoutError):lease.apply(envelope())
    assert session.requests==[(-.3,.2,.1),(0.,0.,1.)]
    assert lease.events[-2]['kind']=='student_request'


def test_release_happens_even_when_stop_submit_fails():
    lease,session,clock=make()
    def fail(*a,**kw):raise RuntimeError('lost ownership')
    session.submit=fail
    with pytest.raises(RuntimeError):lease.release(stationary_feedback_verified=False)
    assert not session.owned and lease.stop_request_failed and lease.state=='released'


def test_unknown_sdk_application_is_recorded_and_never_retried_as_student_control():
    lease,session,clock=make();lease.arm(prepared_since=.4,valid_camera_groups=3)
    submit=session.submit
    def unknown(*a,**kw):
        if kw:raise OSError('driver outcome unknown')
        return submit(*a,**kw)
    session.submit=unknown
    with pytest.raises(OSError):lease.apply(envelope())
    event=next(e for e in lease.events if e['kind']=='student_request')
    assert event['submission_attempted'] and event['application_outcome_unknown']
    assert event['conversion']['requested_control']['throttle']==.2
    assert session.requests==[(0.,0.,1.)] and lease.state=='stopping'


def test_arm_requires_the_new_publication_not_a_still_valid_old_view():
    from flybrain_drive.current_flight_actuation import ready_to_arm
    decision=envelope();preparation=[decision.record() for _ in range(5)]
    kwargs=dict(now=1.02,first_publication=.4,valid_groups=3,publication_id=1,capture_id=4)
    assert ready_to_arm(preparation,decision,**kwargs)
    assert not ready_to_arm(preparation,decision,**dict(kwargs,publication_id=2,capture_id=5))
    assert not ready_to_arm(preparation,replace(decision,view_render_lowers=(.94,)*4),**kwargs)
    assert not ready_to_arm(preparation,replace(decision,body_first_receipt=.99),**kwargs)
    assert not ready_to_arm(preparation,decision,**dict(kwargs,first_publication=.9))
    preparation[-1]['prediction']=[1.,1.,0.]
    assert not ready_to_arm(preparation,decision,**kwargs)
