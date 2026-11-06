import threading
from flybrain_drive.current_flight_dispatcher import CurrentFlightDecisionDispatcher

class Actuator:
    def __init__(self):self.calls=[];self.applied=threading.Event()
    def arm(self,**kw):self.calls.append(('arm',kw))
    def apply(self,value):self.calls.append(('apply',value));self.applied.set();return 'receipt'
    def request(self,kind):self.calls.append((kind,))

class Decision:
    decision_id=1
    def record(self):return {'decision_id':1}

class Owner:
    def __init__(self):self.calls=[]
    def latest_after(self,previous):
        self.calls.append(previous)
        return Decision() if previous==0 else None


def test_dispatcher_consumes_each_decision_once_without_inference():
    owner=Owner();actuator=Actuator();dispatcher=CurrentFlightDecisionDispatcher(owner,actuator)
    dispatcher.start(prepared_since=0,valid_camera_groups=3)
    assert actuator.applied.wait(1)
    dispatcher.stop()
    assert [v[0] for v in actuator.calls]==['arm','apply']
    assert len(dispatcher.decisions)==1 and dispatcher.error is None
    assert owner.calls[0]==0


def test_unknown_owner_reply_causes_stop_and_no_automatic_retry():
    class Lost:
        calls=0
        def latest_after(self,previous):self.calls+=1;raise EOFError('unknown')
    owner=Lost();actuator=Actuator();dispatcher=CurrentFlightDecisionDispatcher(owner,actuator)
    dispatcher.start(prepared_since=0,valid_camera_groups=3);dispatcher.thread.join(1)
    assert not dispatcher.thread.is_alive() and owner.calls==1
    assert dispatcher.error['type']=='EOFError'
    assert [v[0] for v in actuator.calls]==['arm','stop']


def test_readonly_observer_preserves_once_only_decisions_and_has_no_actuator():
    import time
    from flybrain_drive.current_flight_dispatcher import CurrentFlightDecisionObserver
    owner=Owner();observer=CurrentFlightDecisionObserver(owner);observer.start()
    deadline=time.perf_counter()+1
    while not observer.decisions and time.perf_counter()<deadline:time.sleep(.001)
    observer.stop()
    assert len(observer.decisions)==1 and observer.error is None
    assert not hasattr(observer,'actuator')
    assert observer.latest_decision.decision_id==1 and owner.calls.count(0)==1


def test_unknown_apply_retains_full_attempt_without_automatic_resubmission():
    class Unknown(Actuator):
        def apply(self,value):raise TimeoutError('application outcome unknown')
    owner=Owner();actuator=Unknown();dispatcher=CurrentFlightDecisionDispatcher(owner,actuator)
    dispatcher.start(prepared_since=0,valid_camera_groups=3);dispatcher.thread.join(1)
    assert len(dispatcher.attempts)==1 and dispatcher.attempts[0]['decision']=={'decision_id':1}
    assert dispatcher.attempts[0]['status']=='unknown_or_rejected' and dispatcher.decisions==[]
    assert owner.calls==[0] and [v[0] for v in actuator.calls]==['arm','stop']


def test_preparation_observer_preserves_stale_failure_and_never_retries():
    from flybrain_drive.current_flight_dispatcher import CurrentFlightDecisionObserver
    class Stale:
        calls=0
        def latest_after(self,previous):self.calls+=1;raise ValueError('Original body stale')
    owner=Stale();observer=CurrentFlightDecisionObserver(owner);observer.start();observer.thread.join(1);observer.stop()
    assert owner.calls==1 and observer.error['type']=='ValueError'
    assert observer.latest_decision is None and observer.decisions==[]
