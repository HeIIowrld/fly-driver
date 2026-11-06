"""One latest-decision consumer; stop lease on RPC/timing/identity failure."""
import threading,time

class CurrentFlightDecisionDispatcher:
    def __init__(self, owner, actuator):
        self.owner,self.actuator=owner,actuator
        self.stop_event=threading.Event();self.error=None;self.decisions=[];self.attempts=[]
        self.thread=threading.Thread(target=self._run,name='current-flight-decision-dispatch',daemon=True)

    def start(self,*,prepared_since,valid_camera_groups):
        self.actuator.arm(prepared_since=prepared_since,valid_camera_groups=valid_camera_groups)
        self.thread.start()

    def _run(self):
        previous=0
        try:
            while not self.stop_event.is_set():
                decision=self.owner.latest_after(previous)
                if decision is None:time.sleep(.001);continue
                if self.stop_event.is_set():break
                received=time.perf_counter()
                attempt=dict(received=received,decision=decision.record(),status='pending');self.attempts.append(attempt)
                try:receipt=self.actuator.apply(decision)
                except BaseException as error:
                    attempt.update(status='unknown_or_rejected',error=type(error).__name__+': '+str(error));raise
                attempt.update(status='acknowledged',receipt=receipt)
                self.decisions.append(dict(received=received,applied_response_received=time.perf_counter(),decision=decision.record(),receipt=receipt))
                previous=decision.decision_id
        except BaseException as error:
            self.error=dict(type=type(error).__name__,reason=str(error))
            # The independent actuator still expires if the stop RPC is unknown.
            try:self.actuator.request('stop')
            except Exception:pass

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:self.thread.join(1.)
        if self.thread.is_alive():raise RuntimeError('Decision dispatcher remains live')


class CurrentFlightDecisionObserver:
    """Read-only consumer at the actual dispatch cadence; never owns an actuator."""
    def __init__(self,owner,*,read_latest=None):
        self.read_latest=owner.latest_after if read_latest is None else read_latest
        self.stop_event=threading.Event();self.error=None;self.decisions=[];self.latest_decision=None
        self.thread=threading.Thread(target=self._run,name='current-flight-readonly-decisions',daemon=True)
    def start(self):self.thread.start()
    def _run(self):
        previous=0
        try:
            while not self.stop_event.is_set():
                decision=self.read_latest(previous)
                if decision is None:time.sleep(.001);continue
                self.decisions.append(dict(received=time.perf_counter(),decision=decision.record()));self.latest_decision=decision
                previous=decision.decision_id
        except BaseException as error:self.error=dict(type=type(error).__name__,reason=str(error))
    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:self.thread.join(1.)
        if self.thread.is_alive():raise RuntimeError('Read-only decision consumer remains live')
