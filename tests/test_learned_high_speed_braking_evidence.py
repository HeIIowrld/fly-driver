from copy import deepcopy

import pytest

from flybrain_drive.learned_high_speed_braking_evidence import evaluate_braking


def request(t, brake):
    u = dict(steering=.01, throttle=0., brake=brake)
    return dict(kind="student_request", conversion=dict(
        requested_control=u, raw_prediction=[.01/.6,0.,brake],
        pedal_override=False, clipping_applied=False),
        receipt=dict(driver_accepted=True, requested_control=u,
                     driver_call_finished_perf_counter_seconds=t))


def native(i,t,speed,brake,ai=False):
    return dict(packet=i,first_receipt=t,fields=dict(speedKmh=speed,brake=brake),
                guard=dict(numberOfTyresOut=0,carDamage=[0.]*5),authority=dict(builtin_ai_controlled=ai))


def records():
    events=[request(1.,0.),request(1.03,.6),request(1.06,.6),request(1.09,.6),
            dict(kind="stop_request",when=1.13)]
    feedback=[native(1,1.01,110.,0.),native(2,1.045,108.,.6),
              native(3,1.075,105.,.6),native(4,1.105,101.,.6),
              native(5,1.14,80.,1.)]
    return events, feedback


def test_learned_braking_ends_before_cleanup():
    events,feedback=records()
    result=evaluate_braking(events,feedback)
    assert result["passed"] and result["speed_drop_kmh"]==9.
    feedback[2]["fields"]["speedKmh"]=108.
    feedback[3]["fields"]["speedKmh"]=108.
    assert not evaluate_braking(events,feedback)["passed"]


def test_cannot_count_duplicate_native_packet_or_teacher():
    events,feedback=records()
    rows=deepcopy(feedback)
    rows[3]["packet"]=rows[2]["packet"]
    assert not evaluate_braking(events,rows)["passed"]
    feedback[2]["authority"]["builtin_ai_controlled"]=True
    assert evaluate_braking(events,feedback)["reason"]=="native_authority_or_track_limit_failed"


def test_rejects_override_and_no_high_speed():
    events,feedback=records()
    assert not evaluate_braking(events,feedback,entry_speed_kmh=120.)["passed"]
    events[1]["conversion"]["pedal_override"]=True
    with pytest.raises(ValueError,match="Unmodified"):
        evaluate_braking(events,feedback)


def test_new_damage_cannot_supply_the_speed_drop():
    events,feedback=records()
    feedback[3]["guard"]["carDamage"]=[0.,0.,0.,0.,.02]
    assert evaluate_braking(events,feedback)["reason"]=="native_authority_or_track_limit_failed"
