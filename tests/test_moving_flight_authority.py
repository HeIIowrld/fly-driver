import pytest

from flybrain_drive.current_flight_actuation import FlightActuationLease
from flybrain_drive.moving_flight_authority import MovingFlightAuthority
from test_current_flight_control_contract import IDENTITY, envelope


class Sdk:
    def __init__(self):
        self.requests=[]

    def submit(self,steering,throttle,brake):
        self.requests.append((steering,throttle,brake))
        return dict(driver_accepted=True)


def setup():
    clock=[1.02]
    sdk=Sdk()
    lease=FlightActuationLease(sdk,identity=IDENTITY,clock=lambda:clock[0])
    plan=dict(generation="g",epoch=4,track="actual",center_native_xyz_m=[0.,0.,0.],
        forward_xz=[0.,1.],speed_range_kmh=[100.,104.],maximum_along_error_m=2.,
        maximum_lateral_error_m=.5,maximum_direction_error_rad=.1)
    entry=dict(generation="g",epoch=4,track="actual",native_ai=False,tyres_out=0,speed_kmh=102.)
    before=dict(entry,original_position_receipt=.98,packet_id=1,position_native_xyz_m=[0.,0.,-.8])
    after=dict(entry,original_position_receipt=1.01,packet_id=2,position_native_xyz_m=[0.,0.,0.])
    broker=MovingFlightAuthority(lease,pacing_stream="teacher",entry_plan=plan,
        guard=lambda:(before,after),pacing_seconds=3)
    return sdk, broker, clock, after


def pace(broker):
    return broker.submit_pacing(stream="teacher",sequence=1,source_receipt=1.,
                                steering=.1,throttle=.4,brake=0.)


def test_same_handle_revokes_teacher_before_first_unmodified_student_command():
    sdk,b,clock,_=setup()
    pace(b)
    b.grant_student(envelope(),prepared_since=.4,camera_groups=4)
    with pytest.raises(RuntimeError,match="revoked"):
        pace(b)
    assert sdk.requests==[(.1,.4,0.),(-.3,.2,.1)]
    assert [e["kind"] for e in b.lease.events]==[
        "pacing_request","armed","moving_student_grant","student_request"]
    clock[0]=1.051
    b.poll()
    assert b.lease.stop_reason=="source_lease_expired" and sdk.requests[-1]==(0.,0.,1.)


def test_wrong_moving_pose_and_stale_pace_stop_without_student_authority():
    sdk,b,_,after=setup()
    pace(b)
    after["speed_kmh"]=140.
    with pytest.raises(ValueError,match="does not match"):
        b.grant_student(envelope(),prepared_since=.4,camera_groups=4)
    assert sdk.requests==[(.1,.4,0.),(0.,0.,1.)]
    sdk,b,clock,_=setup()
    pace(b)
    clock[0]=1.051
    with pytest.raises(TimeoutError,match="expired"):
        b.grant_student(envelope(),prepared_since=.4,camera_groups=4)
    assert len(sdk.requests)==2 and b.state=="terminal"


def test_slow_final_native_check_cannot_extend_previous_pacing_request():
    sdk,b,clock,_=setup()
    pace(b)
    old=b.guard
    checks=[0]
    def slow():
        checks[0]+=1
        if checks[0]==2:clock[0]=1.051
        return old()
    b.guard=slow
    with pytest.raises(TimeoutError,match="before first student"):
        b.grant_student(envelope(),prepared_since=.4,camera_groups=4)
    assert sdk.requests==[(.1,.4,0.),(0.,0.,1.)]


def test_expired_unarmed_observation_clears_readiness_but_not_current_teacher_request(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import Mock

    from flybrain_drive.moving_flight_trial import MovingStudentTrial

    original = envelope()
    mailbox = Mock()
    mailbox.read.return_value = original
    broker = Mock()
    trial = SimpleNamespace(mailbox=mailbox, broker=broker, lease=SimpleNamespace(identity=IDENTITY),
                            pacing_stream="original_sac",
                            publications={1: original.published_at}, last_expired_publication=None, emit=Mock())
    monkeypatch.setattr("flybrain_drive.moving_flight_trial.time.perf_counter", lambda: 1.051)
    MovingStudentTrial.pacing(trial, (.1, .3, 0.), 2, 1.03)
    assert trial.publications == {}
    assert trial.last_expired_publication == 1
    assert trial.emit.call_args.args == ("moving_unarmed_decision_expired",)
    broker.submit_pacing.assert_called_once_with(stream="original_sac", sequence=2,
        source_receipt=1.03, steering=.1, throttle=.3, brake=0.)

    mailbox.read.return_value = replace(original, generation="changed")
    with pytest.raises(ValueError, match="identity"):
        MovingStudentTrial.pacing(trial, (.1, .3, 0.), 3, 1.03)
    assert broker.submit_pacing.call_count == 1
