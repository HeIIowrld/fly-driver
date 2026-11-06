import math
import numpy as np
import pytest
from flybrain_drive.fine_motion_history import FineMotionHistory


def make():
    h=FineMotionHistory(np.zeros(2048),np.ones(2048));h.begin_episode('test');return h


def step(h,value,t,c=None,clocks=None):
    c=t-.01 if c is None else c;clocks=tuple(c-.004+j*.001 for j in range(4)) if clocks is None else clocks
    return h.step(np.full(2048,value,np.float32),episode_id='test',decision_time=t,camera_receipt=c,render_lowers=clocks)


def test_new_observation_cannot_retroactively_change_prior_interval():
    h=make();first=step(h,0.,1.);second=step(h,1.,1.04);assert np.count_nonzero(first['innovation'])==0;assert np.all(second['innovation']==1) and np.count_nonzero(second['fast_minus_slow'])==0
    old=h.state;third=step(h,1.,1.06,c=old.camera_receipt,clocks=old.render_lowers);np.testing.assert_allclose(third['innovation'],math.exp(-.02/.2),atol=1e-7);np.testing.assert_allclose(third['fast_minus_slow'],math.exp(-.02/.2)-math.exp(-.02/.05),atol=1e-7)


@pytest.mark.parametrize('kind',['mutated_frame','reverse_camera','decision_gap','stale_view','wrong_episode'])
def test_bad_input_is_terminal_without_history_rewind(kind):
    h=make();step(h,0.,1.);old=h.state
    with pytest.raises(ValueError):
        if kind=='mutated_frame':step(h,1.,1.02,c=old.camera_receipt,clocks=old.render_lowers)
        if kind=='reverse_camera':step(h,1.,1.02,c=.98)
        if kind=='decision_gap':step(h,1.,1.1)
        if kind=='stale_view':step(h,1.,1.02,c=.8)
        if kind=='wrong_episode':h.step(np.zeros(2048),episode_id='other',decision_time=1.02,camera_receipt=1.01,render_lowers=(1.,1.001,1.002,1.003))
    assert h.poisoned and h.state is old
    with pytest.raises(RuntimeError):step(h,0.,1.04)


def test_explicit_episode_restart_resets_and_state_is_readonly():
    h=make();step(h,0.,1.);step(h,1.,1.04)
    with pytest.raises(ValueError):h.state.held[0]=99
    h.begin_episode('test');result=step(h,2.,2.);assert np.count_nonzero(result['innovation'])==0


def test_nonfinite_filter_time_constant_is_rejected():
    with pytest.raises(ValueError):FineMotionHistory(np.zeros(2048),np.ones(2048),slow_seconds=float('inf'))
