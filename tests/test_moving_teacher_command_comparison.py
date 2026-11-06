import pytest
import numpy as np
from flybrain_drive.moving_teacher_command_comparison import accepted_teacher_commands,pair_past_commands,discrepancy


def event(finished=1.,steering=-.3):
    return dict(kind='teacher_request',stage=0,decision_started_host_time=finished-.02,request=dict(driver_accepted=True,game_applied=False,driver_call_started_perf_counter_seconds=finished-.01,driver_call_finished_perf_counter_seconds=finished,requested_control=dict(steering=steering,throttle=.2,brake=.4)))


def test_scale_matches_training_target_and_does_not_claim_application():
    commands=accepted_teacher_commands([event()],.6)
    assert commands[0]['target']==[-.5,.2,.4] and not commands[0]['game_applied_verified']


def test_future_command_is_not_used_and_stale_label_remains_missing():
    commands=accepted_teacher_commands([event(),event(2.,.3)],.6)
    pairs=pair_past_commands([.99,1.,1.05,1.2,1.99,2.],commands,maximum_age_seconds=.06)
    assert [p['eligible'] for p in pairs]==[False,True,True,False,False,True]
    assert pairs[4]['command_index']==0 and pairs[5]['command_index']==1


def test_normal_commands_exclude_terminal_stop_targets():
    stop=event(1.1);stop['kind']='teacher_stop_request'
    assert len(accepted_teacher_commands([event(),stop],.6))==1


@pytest.mark.parametrize('problem',['unaccepted','future_start','duplicate','out_of_bounds'])
def test_invalid_authority_or_units_are_rejected(problem):
    e=event();events=[e]
    if problem=='unaccepted':e['request']['driver_accepted']=False
    if problem=='future_start':e['request']['driver_call_started_perf_counter_seconds']=2.
    if problem=='duplicate':events=[e,event()]
    if problem=='out_of_bounds':e['request']['requested_control']['steering']=.7
    with pytest.raises(ValueError):accepted_teacher_commands(events,.6)


def test_repeated_body_ticks_do_not_overweight_command_balanced_metric():
    p=np.array([[1,0,0],[1,0,0],[0,0,0]]);y=np.zeros_like(p)
    result=discrepancy(p,y,[0,0,1])
    assert result['rmse'][0]==pytest.approx(np.sqrt(2/3))
    assert result['command_balanced_rmse'][0]==pytest.approx(np.sqrt(.5))
    assert result['unique_teacher_commands']==2


def test_opposite_sign_requires_material_nonzero_steering():
    r=discrepancy([[.3,.2,.2],[-.01,0,0],[-.5,0,0]],[[-.3,0,0],[.3,0,0],[-.3,0,0]],[0,1,2])
    assert r['material_steering_events']==2 and r['opposite_steering_events']==1 and r['simultaneous_pedals_above_point1']==1


@pytest.mark.parametrize('times',[[1.,1.],[2.,1.],[float('nan')],[]])
def test_reversed_or_invalid_decisions_rejected(times):
    with pytest.raises(ValueError):pair_past_commands(times,accepted_teacher_commands([event()],.6),maximum_age_seconds=.06)
