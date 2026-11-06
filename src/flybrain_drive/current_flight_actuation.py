"""Finite fail-closed actuator lease. No inference, teacher or route enters this core."""
import math
from .current_flight_control_contract import control_request


class FlightActuationLease:
    def __init__(self, session, *, identity, clock, maximum_seconds=2., maximum_decisions=100):
        if not 0 < maximum_seconds <= 5 or type(maximum_decisions) is not int or not 1 <= maximum_decisions <= 500:
            raise ValueError("Finite trial bounds required")
        self.session, self.identity, self.clock = session, dict(identity), clock
        self.maximum_seconds, self.maximum_decisions = maximum_seconds, maximum_decisions
        self.state = 'preparing'
        self.last_decision = self.last_publication = 0
        self.last_capture = -1
        self.started = self.expires = None
        self.events = []
        self.stop_reason = None
        self.stop_request_failed = False

    def arm(self, *, prepared_since, valid_camera_groups):
        now = self.clock()
        if (self.state != 'preparing' or not math.isfinite(prepared_since)
                or not .5 <= now-prepared_since <= 10
                or type(valid_camera_groups) is not int or valid_camera_groups < 3):
            raise ValueError("Continuous half-second/three-group readiness required")
        self.state = 'armed'
        self.started = now
        # No coast/drive between arming and first valid decision.
        self.expires = now + .05
        self.events.append(dict(kind='armed',when=now,prepared_since=prepared_since,valid_camera_groups=valid_camera_groups))

    def apply(self, decision):
        try:
            self.poll()
            if self.state != 'armed':raise RuntimeError("Actuation is terminal or not armed")
            now = self.clock()
            expires = decision.validate(now=now, **self.identity, after_decision=self.last_decision,
                                        after_publication=self.last_publication, after_capture=self.last_capture)
            if self.last_decision and ((decision.publication_id == self.last_publication) != (decision.capture_id == self.last_capture)):
                raise ValueError("Publication and capture identity disagree")
            request = control_request(decision.prediction)
            event=dict(kind='student_request',when=now,decision=decision.record(),conversion=request,submission_attempted=True)
            self.events.append(event)
            try:
                receipt = self.session.submit(**request['requested_control'])
            except BaseException as error:
                event.update(finished=self.clock(),error=type(error).__name__+': '+str(error),application_outcome_unknown=True)
                raise
            finished = self.clock()
            event.update(finished=finished,receipt=receipt)
            self.last_decision, self.last_publication, self.last_capture = decision.decision_id, decision.publication_id, decision.capture_id
            self.expires = expires
            self.applied = getattr(self, 'applied', 0) + 1
            if finished > expires:raise TimeoutError("SDK request finished after source deadline")
            if self.applied >= self.maximum_decisions:self.stop('decision_budget')
            return receipt
        except BaseException as error:
            self.stop(type(error).__name__+': '+str(error))
            raise

    def poll(self):
        if self.state == 'armed':
            now = self.clock()
            if now >= self.expires:self.stop('source_lease_expired')
            elif now-self.started >= self.maximum_seconds:self.stop('trial_time_budget')

    def stop(self, reason):
        if self.state in ('stopping','released'):return
        self.state = 'stopping'
        self.stop_reason = reason
        event = dict(kind='stop_request',when=self.clock(),reason=reason,cleanup_not_learned_braking=True)
        self.events.append(event)
        try:event['receipt'] = self.session.submit(0.,0.,1.)
        except BaseException as error:
            self.stop_request_failed = True
            event['error'] = type(error).__name__+': '+str(error)

    def release(self, *, stationary_feedback_verified):
        self.stop(self.stop_reason or 'explicit_stop')
        try:
            receipt = self.session.close()
            self.events.append(dict(kind='release',when=self.clock(),receipt=receipt,
                                    stationary_feedback_verified=stationary_feedback_verified))
        finally:self.state = 'released'


def ready_to_arm(preparation,decision,*,now,first_publication,valid_groups,publication_id,capture_id):
    """Readiness reserves are stricter than the unchanged application deadlines."""
    if (decision is None or first_publication is None or now-first_publication<.5
            or valid_groups<3 or len(preparation)<5
            or decision.publication_id!=publication_id or decision.capture_id!=capture_id
            or not 0<=now-decision.body_first_receipt<=.025
            or not 0<=now-min(decision.view_render_lowers)<=.075):
        return False
    import numpy as np
    return bool(np.ptp(np.asarray([v['prediction'] for v in preparation[-5:]]),axis=0).max()<=.05)
