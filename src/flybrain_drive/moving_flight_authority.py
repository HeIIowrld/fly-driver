"""Single-owner pacing-to-student transfer core for future moving trials.

The caller must supply independently validated native guards and run this on
the SDK owner's serial loop. No device discovery, lifecycle or actor input.
"""
import math

from .current_flight_actuation import FlightActuationLease
from .matched_corner_entry import corner_entry_measurement


class MovingFlightAuthority:
    def __init__(self, lease, *, pacing_stream, entry_plan, guard, pacing_seconds):
        if not isinstance(lease, FlightActuationLease) or lease.state != "preparing":
            raise ValueError("Fresh single-owner actuation lease required")
        if (not isinstance(pacing_stream,str) or not pacing_stream or not callable(guard)
                or not math.isfinite(pacing_seconds) or not 0 < pacing_seconds <= 180):
            raise ValueError("Explicit owned pacing stream and native guard required")
        if entry_plan["generation"]!=lease.identity["generation"]:
            raise ValueError("Pacing and student must share the same session generation")
        self.lease=lease
        self.pacing_stream=pacing_stream
        self.plan=dict(entry_plan)
        self.guard=guard
        self.state="pacing"
        self.pace_sequence=0
        self.pacing_deadline=None
        self.entry_deadline=lease.clock()+pacing_seconds

    def submit_pacing(self, *, stream, sequence, source_receipt, steering, throttle, brake):
        if self.state!="pacing":
            raise RuntimeError("Pacing authority was permanently revoked")
        now=self.lease.clock()
        try:
            if (now>=self.entry_deadline or stream!=self.pacing_stream or type(sequence) is not int or sequence<=self.pace_sequence
                    or not math.isfinite(source_receipt) or not 0<=now-source_receipt<.05
                    or any(not math.isfinite(x) for x in (steering,throttle,brake))
                    or not -1<=steering<=1 or not 0<=throttle<=1 or not 0<=brake<=1):
                raise ValueError("Fresh original bounded pacing request required")
            self.guard()
            if self.lease.clock()>=min(source_receipt+.05,self.entry_deadline):
                raise TimeoutError("Pacing source expired during guard verification")
            event=dict(kind="pacing_request",source_receipt=source_receipt,when=self.lease.clock(),
                       sequence=sequence,request=dict(steering=steering,throttle=throttle,brake=brake))
            self.lease.events.append(event)
            receipt=self.lease.session.submit(steering,throttle,brake)
            event["receipt"]=receipt
            if receipt.get("driver_accepted") is not True or self.lease.clock()>=min(source_receipt+.05,self.entry_deadline):
                raise TimeoutError("Pacing SDK submission not verified within original source deadline")
            self.pace_sequence=sequence
            self.pacing_deadline=source_receipt+.05
            return receipt
        except BaseException:
            self.stop("moving_pacing_fault")
            raise

    def grant_student(self, decision, *, prepared_since, camera_groups):
        if self.state!="pacing":
            raise RuntimeError("Moving student authority is single-grant")
        try:
            if self.pacing_deadline is None or self.lease.clock()>=self.pacing_deadline:
                raise TimeoutError("Original pacing authority already expired")
            previous,current=self.guard()
            now=self.lease.clock()
            entry=corner_entry_measurement(self.plan,previous,current,now=now)
            if not entry["eligible"]:
                raise ValueError("Actual moving entry does not match the qualified pose and speed")
            decision.validate(now=now,**self.lease.identity)
            if now>=min(self.pacing_deadline,self.entry_deadline):
                raise TimeoutError("Original pacing authority expired during entry admission")
            self.lease.arm(prepared_since=prepared_since,valid_camera_groups=camera_groups)
            self.state="student"
            self.lease.events.append(dict(kind="moving_student_grant",when=self.lease.clock(),
                last_pacing_sequence=self.pace_sequence,entry=entry,first_student_decision=decision.decision_id))
            return self._submit_student(decision, first_deadline=min(self.pacing_deadline,self.entry_deadline))
        except BaseException:
            self.stop("moving_entry_fault")
            raise

    def submit_student(self,decision):
        return self._submit_student(decision)

    def _submit_student(self,decision,*,first_deadline=None):
        if self.state!="student":
            raise RuntimeError("Student does not own this SDK session")
        try:
            self.guard()
            if first_deadline is not None and self.lease.clock()>=first_deadline:
                raise TimeoutError("Prior pacing request expired before first student SDK command")
            receipt=self.lease.apply(decision)
            if receipt.get("driver_accepted") is not True:
                raise RuntimeError("Student driver acceptance unverified")
            if self.lease.state!="armed":self.state="terminal"
            return receipt
        except BaseException:
            self.stop("moving_student_fault")
            raise

    def poll(self):
        if self.state=="pacing":
            now=self.lease.clock()
            if now>=self.entry_deadline:self.stop("moving_entry_budget")
            elif self.pacing_deadline is not None and now>=self.pacing_deadline:self.stop("moving_pacing_source_expired")
        elif self.state=="student":
            self.lease.poll()
            if self.lease.state!="armed":self.state="terminal"

    def stop(self,reason):
        self.state="terminal"
        self.lease.stop(reason)
