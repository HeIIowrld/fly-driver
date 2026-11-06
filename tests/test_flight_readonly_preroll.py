import pytest
from flybrain_drive.flight_readonly_preroll import ReadOnlyFlightPreroll

class Runtime:
    episode='one';last_time=None;closed=False;poisoned=False
    def __init__(self):self.steps=0
    def decide(self,*,decision_time):
        if self.poisoned:raise ValueError('source rejected')
        self.steps+=1;self.last_time=decision_time
        return dict(prediction=self.steps,actor_qualified=False,vehicle_control_sent=False)


def test_preroll_keeps_same_state_across_preparation_boundary():
    runtime=Runtime();gate=ReadOnlyFlightPreroll(runtime,minimum_seconds=.5)
    assert gate.decide(capture_id=1,decision_time=1.)['continuation_candidate'] is None
    assert gate.decide(capture_id=2,decision_time=1.25)['continuation_candidate'] is None
    assert gate.decide(capture_id=3,decision_time=1.5)['continuation_candidate']==3
    assert gate.decide(capture_id=3,decision_time=1.51)['continuation_candidate']==4
    assert runtime.steps==4 and gate.events[-1]['distinct_captures']==3
    assert not gate.events[-1]['actor_qualified']


def test_time_alone_does_not_complete_visual_preparation():
    runtime=Runtime();gate=ReadOnlyFlightPreroll(runtime,minimum_seconds=.5)
    gate.decide(capture_id=1,decision_time=1.)
    assert gate.decide(capture_id=1,decision_time=2.)['continuation_candidate'] is None


@pytest.mark.parametrize('problem',['reset','external_step','camera_reverse','source'])
def test_reset_or_source_failure_is_terminal_without_resetting_state(problem):
    runtime=Runtime();gate=ReadOnlyFlightPreroll(runtime,minimum_seconds=.5);gate.decide(capture_id=2,decision_time=1.)
    if problem=='reset':runtime.last_time=None
    if problem=='external_step':runtime.last_time=1.1
    if problem=='source':runtime.poisoned=True
    with pytest.raises(ValueError):gate.decide(capture_id=1 if problem=='camera_reverse' else 3,decision_time=1.5)
    assert runtime.steps==1
    with pytest.raises(RuntimeError):gate.decide(capture_id=4,decision_time=2.)


@pytest.mark.parametrize('seconds,captures',[(0,3),(float('nan'),3),(True,3),(.5,1),(.5,True)])
def test_invalid_preparation_contract_rejected(seconds,captures):
    with pytest.raises(ValueError):ReadOnlyFlightPreroll(Runtime(),minimum_seconds=seconds,minimum_captures=captures)
