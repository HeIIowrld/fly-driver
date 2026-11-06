from types import SimpleNamespace
import pytest
from flybrain_drive.live_thermal_resume_readiness import wait_live_stationary


def harness(speeds,packets=None,*,terminal_status='api_returned_true',state_change=None,ai=False,delayed_heartbeat=False,prepared=False):
    request=dict(id='r',command='resume');before=dict(generation='g',epoch=1)
    base=dict(generation='g',epoch=1,script_clock=1.,car_id='bmw_z4_gt3',track_id='monza',
        is_online=False,in_menu=False,ai_controlled=False,camera_active=False,expert_probe_active=False,
        brake_settle_active=False,physics_api_allowed=True,paused=False,live=True,speed_kmh=0.,gas=0.,brake=1.)
    if state_change:base.update(state_change)
    index=[0];clock=[0.];records=[]
    def native():
        k=min(index[0],len(speeds)-1);index[0]+=1
        packet=packets[k] if packets else k+11
        return SimpleNamespace(builtin_ai_controlled=ai,
            physics=SimpleNamespace(packet_id=packet,received_monotonic_seconds=float(index[0]),fields=dict(speedKmh=speeds[k],gas=0.,brake=1.)),
            authority_record=lambda:dict(packet_id=packet,received_monotonic_seconds=float(index[0]),original_speed=speeds[k]))
    def now():clock[0]+=.05;return clock[0]
    def read_state():
        result=dict(base)
        if delayed_heartbeat and index[0]<2:result.update(paused=True,live=False)
        return result
    def receipt():
        result=dict(id='r',command='resume',status=terminal_status)
        if delayed_heartbeat:result['after']=dict(base,paused=True,live=False,script_clock=.9)
        return result
    run=lambda:wait_live_stationary(request,before,10,read_state=read_state,read_native=native,
        poll_terminal=receipt,check_ownership=lambda s:None,
        observations=records,clock=now,sleep=lambda t:None,initialized_thermal=prepared)
    return run,records


def test_initial_motion_is_retained_and_three_new_stationary_packets_are_required():
    run,records=harness([1.83,.2,0.,0.,0.])
    state,native,witnesses,terminal=run()
    assert [r['native']['original_speed'] for r in records]==[1.83,.2,0.,0.,0.]
    assert [w['packet_id'] for w in witnesses]==[13,14,15]
    assert not any(r['actor_admitted'] for r in records)
    assert [r['ready_native'] for r in records]==[False,False,True,True,True]


def test_duplicates_cannot_manufacture_readiness():
    run,records=harness([0.,0.,0.,0.],[11,11,11,11])
    with pytest.raises(TimeoutError):run()
    assert len(records)>3


@pytest.mark.parametrize('change',[dict(ai_controlled=True),dict(camera_active=True),dict(epoch=2),dict(generation='other'),dict(pending_receipt_id='other'),dict(gas=.1),dict(speed_kmh=6.)])
def test_context_loss_aborts_instead_of_waiting_through_it(change):
    run,records=harness([0.,0.,0.],state_change=change)
    with pytest.raises(ValueError):run()


@pytest.mark.parametrize('problem',['native_ai','fast_native','packet_reset','api_failure'])
def test_unsafe_native_or_unknown_resume_never_produces_ready_witnesses(problem):
    kwargs={}
    if problem=='native_ai':kwargs['ai']=True
    if problem=='packet_reset':kwargs['packets']=[9,10,11]
    if problem=='api_failure':kwargs['terminal_status']='api_returned_false'
    run,records=harness([6.,0.,0.] if problem=='fast_native' else [0.,0.,0.],**kwargs)
    with pytest.raises(ValueError):run()


def test_immediate_api_after_state_may_still_be_paused_until_observed_transition():
    run,records=harness([0.,0.,0.,0.,0.],delayed_heartbeat=True)
    current,native,witnesses,terminal=run()
    assert terminal['after']['paused'] is True
    assert current['paused'] is False and current['live'] is True
    assert [r['ready_native'] for r in records[:2]]==[False,False]
    assert len(witnesses)==3


def test_initialized_gear_resume_requires_exact_consumed_thermal_epoch():
    run,_=harness([0.,0.,0.],state_change={'tyres_initialization_epoch':1},prepared=True)
    _,_,witnesses,_=run()
    assert len(witnesses)==3
    run,_=harness([0.,0.,0.],state_change={'tyres_initialization_epoch':1})
    with pytest.raises(ValueError):run()
    run,_=harness([0.,0.,0.],prepared=True)
    with pytest.raises(ValueError):run()
