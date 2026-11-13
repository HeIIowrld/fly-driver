"""One bounded moving student interval inside the existing teacher SDK owner."""
import math
import time
from pathlib import Path

from .ac_teacher_physics import NativeTeacherPhysicsReader, repeated_teacher_snapshot
from .current_flight_actuation import FlightActuationLease
from .evidence import write_json
from .flight_decision_mailbox import FlightDecisionMailbox
from .graph import sha256
from .learned_high_speed_braking_evidence import evaluate_braking
from .magione_stage import MAGIONE_MOVING_STAGE_PROFILES
from .matched_corner_entry import native_motion_resolved
from .moving_flight_authority import MovingFlightAuthority
from .moving_stop_calibration import MONZA_GEOMETRY_PATH, MONZA_TEACHER_STOP_PROFILES, native_track_fields, validate_natural_monza_reference
from .sac_unified_student import read


def admit_calibration(reference_path, audit_path):
    audit_path = Path(audit_path)
    audit = read(audit_path)
    config = read(audit_path.parent/"manifest.json")["config"]
    root = Path(config["args"]["recording"])
    reference = read(reference_path)
    profile = reference.get("teacher_pacing_profile")
    if profile in MONZA_TEACHER_STOP_PROFILES:
        raise ValueError("Upstream teacher stop cannot authorize student control")
    magione = profile in MAGIONE_MOVING_STAGE_PROFILES
    natural = not magione and validate_natural_monza_reference(reference)
    teacher = root if magione else root/"teacher"
    terminal = read(teacher/"result.json")
    teacher_cfg = read(teacher/"manifest.json")["config"]
    if magione and (teacher_cfg.get("expected_track") != "magione"
            or config.get("geometry_result_sha256") != reference.get("geometry_result_sha256")
            or teacher_cfg.get("geometry_result_sha256") != reference.get("geometry_result_sha256")
            or not audit.get("entry_forward_native_xz") or audit.get("signed_minimum_center_to_border_m",0)<1.8):
        raise ValueError("Exact audited Magione route and moving entry position required")
    if (audit["status"]!="completed_stop_audit" or audit["moving_stop_calibration_passed"] is not True
            or audit["stationary_verified"] is not True or audit["maximum_tyres_out"] != 0
            or sha256(teacher/"result.json") != config["teacher_result_sha256"]
            or sha256(reference_path) != teacher_cfg["moving_stop_reference_sha256"]
            or terminal["moving_stop_calibration"]["new_damage"] is not False
            or (reference.get("pacing_only") is not True and not natural)):
        raise ValueError("Verified same-reference on-track owner stop required before moving student grant")
    speed = audit["entry_speed_kmh"]
    distance = audit["stop_distance_m"]
    reserve = reference["straight_length_m"]-distance
    measured_margin = audit["unsigned_minimum_center_to_border_m"]
    if natural:
        if config.get("geometry_result_sha256") != sha256(MONZA_GEOMETRY_PATH):
            raise ValueError("Natural Monza stop must use the pinned route geometry")
        measured_margin = audit.get("signed_minimum_center_to_border_m", 0.)
        entry = audit.get("entry_native_position") or {}
        coordinates = entry.get("coordinates_native_xyz_m")
        heading = audit.get("entry_forward_native_xz")
        forward = reference["forward_native_xz"]
        brake = audit.get("entry_native_brake", math.nan)
        if not coordinates or not heading:
            raise ValueError("Natural Monza stop entry pose or direction missing")
        cx, _, cz = reference["native_center_xyz_m"]
        x, _, z = coordinates
        fx, fz = forward
        along = (x - cx) * fx + (z - cz) * fz
        lateral = (z - cz) * fx - (x - cx) * fz
        if (not 0. <= brake <= 1. or (brake > .1) != (reference["approach_phase"] == "active_braking")
                or not abs(along) <= 2.5 or not abs(lateral) <= .55
                or not math.dist(heading, forward) <= 2 * math.sin(.045 / 2)
                or not abs(speed - reference["approach"]["speed_kmh"]) <= 4):
            raise ValueError("Natural Monza stop does not match the screened entry")
    margin = min(reference["predicted_minimum_center_clearance_m"], measured_margin)
    minimum_speed, maximum_speed = (24, 30) if magione else (100, 145)
    if profile == "magione_alternative_corner_30_v1":
        minimum_speed, maximum_speed = 26.8, 28.
    if not (minimum_speed<=speed<=maximum_speed and distance>0 and reserve>=16 and margin>=1.8):
        raise ValueError("Insufficient measured moving stop reserve or clearance")
    return reference,audit


class MovingStudentTrial:
    def __init__(self, *, plan, reference_path, audit_path, generation, epoch, request_id,
                 calibration, session, guard, emit):
        reference,audit=admit_calibration(reference_path,audit_path)
        identity=plan["identity"]
        if (plan.get("schema")!="flybrain.moving_student_trial.v1"
                or identity["generation"]!=generation or identity["episode_id"]!=request_id
                or plan["reference_sha256"]!=sha256(reference_path) or plan["audit_sha256"]!=sha256(audit_path)):
            raise ValueError("Exact scoped moving student plan required")
        self.guard,self.emit,self.session=guard,emit,session
        self.terminal_path=Path(plan["terminal_path"])
        if self.terminal_path.exists():
            raise ValueError("Fresh owned moving interval terminal path required")
        self.calibration=calibration
        profile = reference.get("teacher_pacing_profile")
        magione = profile in MAGIONE_MOVING_STAGE_PROFILES
        upper = audit["entry_speed_kmh"]
        self.maximum_calibrated_speed_kmh = upper
        center = audit["entry_native_position"]["coordinates_native_xyz_m"] if magione else reference["native_center_xyz_m"]
        forward = audit["entry_forward_native_xz"] if magione else reference["forward_native_xz"]
        speeds = [max(21,upper-3),upper] if magione else [max(100,upper-4),upper]
        if profile == "magione_alternative_corner_30_v1":
            speeds = [21.5, upper-4.65]
        self.entry=dict(generation=generation,epoch=epoch,track="magione" if magione else "monza",
            center_native_xyz_m=center,forward_xz=forward,speed_range_kmh=speeds,
            maximum_along_error_m=1.2 if magione else 2.5,
            maximum_lateral_error_m=.55,maximum_direction_error_rad=.045)
        if magione:
            self.calibration.entry_plan = self.entry
        self.duration=.35
        self.travel_limit=min(15.,reference["straight_length_m"]-audit["stop_distance_m"]-5.)
        self.lease=FlightActuationLease(session,identity=identity,clock=time.perf_counter,
            maximum_seconds=.45,maximum_decisions=100)
        self.reader=NativeTeacherPhysicsReader(timeout=.05)
        try:self.mailbox=FlightDecisionMailbox(plan["mailbox"])
        except BaseException:
            self.reader.close()
            raise
        self.pacing_stream = "geometric_magione" if magione else "original_sac"
        self.broker=MovingFlightAuthority(self.lease,pacing_stream=self.pacing_stream,
            entry_plan=self.entry,guard=self._guard,pacing_seconds=85.)
        self.feedback=[]
        self.latest=self.previous=None
        self.publications={}
        self.last_expired_publication=None
        self.active=False
        self.start=None
        self.damage=None

    def _guard(self):
        self.guard()
        if self.mailbox.stop_requested:raise RuntimeError("Owned student publisher requested stop")
        if not self.active:return None
        coupled=repeated_teacher_snapshot(self.reader.mapping)
        sample=coupled.physics
        position=self.calibration.positions.snapshot(maximum_age=.05)
        tyres,damage=native_track_fields(sample)
        if position.status!="live" or coupled.builtin_ai_controlled or tyres or any(
                v>base+.01 for v,base in zip(damage,self.damage)):
            raise ValueError("Moving native authority, tyres or damage limit failed")
        stamp=sample.received_monotonic_seconds
        if not self.feedback or sample.packet_id!=self.feedback[-1]["packet"]:
            self.feedback.append(dict(packet=sample.packet_id,first_receipt=stamp,fields=dict(sample.fields),
                guard=dict(numberOfTyresOut=tyres,carDamage=damage),authority=coupled.authority_record(),
                diagnostic_position=position.metadata()))
        elif stamp-self.feedback[-1]["first_receipt"]>.05:
            raise TimeoutError("Moving native body packet stopped advancing")
        point=position.coordinates_native_xyz_m
        row=dict(generation=self.entry["generation"],epoch=self.entry["epoch"],track=self.entry["track"],
            native_ai=False,tyres_out=tyres,speed_kmh=sample.fields["speedKmh"],
            packet_id=position.packet_id,original_position_receipt=position.received_perf_counter_seconds,
            position_native_xyz_m=point)
        if self.latest is None or row["packet_id"]>self.latest["packet_id"]:
            if self.latest is not None and native_motion_resolved(
                    self.latest["position_native_xyz_m"], point):
                self.previous = self.latest
            self.latest = row
        cx,_,cz=self.entry["center_native_xyz_m"]
        tx,tz=self.entry["forward_xz"]
        along=(point[0]-cx)*tx+(point[2]-cz)*tz
        side=(point[2]-cz)*tx-(point[0]-cx)*tz
        if (abs(side)>.55 or along>self.travel_limit or row["speed_kmh"]>self.maximum_calibrated_speed_kmh):
            raise ValueError("Moving stop reserve boundary reached")
        if self.previous is not None:
            a,b=self.previous["position_native_xyz_m"],self.latest["position_native_xyz_m"]
            elapsed = self.latest["original_position_receipt"]-self.previous["original_position_receipt"]
            if not 0 < elapsed <= .05 or not native_motion_resolved(a, b):
                raise ValueError("Fresh moving native heading is unavailable")
            heading=math.atan2((b[2]-a[2])*tx-(b[0]-a[0])*tz,
                               (b[0]-a[0])*tx+(b[2]-a[2])*tz)
            if abs(heading)>self.entry["maximum_direction_error_rad"]:
                raise ValueError("Moving heading left verified straight-stop reserve")
        return self.previous,self.latest

    def pacing(self, controls, sequence, original_receipt):
        # The original teacher renews from a new packet on its 40ms loop.
        # Check its current input here and the latest receipt at the actual transfer.
        decision=self.mailbox.read()
        if decision is not None:
            now=time.perf_counter()
            try:
                decision.validate(now=now,**self.lease.identity)
            except ValueError as error:
                body_age=now-decision.body_first_receipt
                view_age=now-min(decision.view_render_lowers)
                if (str(error)!="Decision body50ms/view150ms freshness/order exceeded"
                        or body_age<=.05 and view_age<=.15):
                    raise
                # The student has no authority during pacing. Forget readiness,
                # keep the actor's recurrence, and do not promote this decision.
                self.publications.clear()
                if decision.publication_id!=self.last_expired_publication:
                    self.emit("moving_unarmed_decision_expired",decision=decision.record(),
                        observed=now,body_age_seconds=body_age,view_age_seconds=view_age)
                    self.last_expired_publication=decision.publication_id
            else:
                self.publications.setdefault(decision.publication_id,decision.published_at)
        return self.broker.submit_pacing(stream=self.pacing_stream,sequence=sequence,source_receipt=original_receipt,
            steering=float(controls[0]),throttle=float(controls[1]),brake=float(controls[2]))

    def run(self, gate, *, reader):
        self.active=True
        pos,sample,self.damage=gate
        before=self.calibration.previous_graphics
        if before is not None:
            self.previous=dict(self.entry,native_ai=False,tyres_out=0,speed_kmh=sample.fields["speedKmh"],
                packet_id=before.packet_id,original_position_receipt=before.received_perf_counter_seconds,
                position_native_xyz_m=before.coordinates_native_xyz_m)
        self.latest=dict(self.entry,native_ai=False,tyres_out=0,speed_kmh=sample.fields["speedKmh"],
            packet_id=pos.packet_id,original_position_receipt=pos.received_perf_counter_seconds,
            position_native_xyz_m=pos.coordinates_native_xyz_m)
        error=None
        try:
            d=self.mailbox.read()
            now=time.perf_counter()
            recent=[stamp for stamp in self.publications.values() if 0 <= now-stamp <= 2.]
            if d is None or len(recent)<3 or now-min(recent)<.5:
                raise ValueError("Fresh uninterrupted student prefix required")
            self.broker.grant_student(d,prepared_since=min(recent),camera_groups=len(recent))
            self.start=self.lease.started
            while self.broker.state=="student":
                self.broker.poll()
                if self.broker.state!="student":break
                if time.perf_counter()-self.start>=self.duration:
                    self.broker.stop("planned_observation_complete")
                    break
                self._guard()
                d=self.mailbox.read(self.lease.last_decision)
                if d is not None:self.broker.submit_student(d)
                time.sleep(.001)
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            self.broker.stop("moving_student_or_entry_fault")
        finally:
            self.active=False
            write_json(self.terminal_path,dict(episode_id=self.lease.identity["episode_id"],
                moving_interval_ended=True,stop_reason=self.lease.stop_reason))
            for event in self.lease.events:
                if event["kind"]!="pacing_request":
                    self.emit("moving_actuator_event",event=event)
            for sample in self.feedback:self.emit("moving_student_feedback",sample=sample)
        stops=[v for v in self.lease.events if v["kind"]=="stop_request"]
        if not stops or "receipt" not in stops[-1]:
            raise RuntimeError("Moving stop request not verified")
        stopped_position=self.calibration.positions.snapshot(maximum_age=.08)
        cleanup=self.calibration.watch_stop((stopped_position,sample,self.damage),stops[-1]["receipt"],
            reader=reader,guard=self.guard,emit=self.emit)
        audit=evaluate_braking(self.lease.events,self.feedback)
        return dict(status="completed_moving_interval" if error is None and cleanup["status"]=="completed_stop_measurement"
                    and self.lease.stop_reason=="planned_observation_complete"
                    else "failed_moving_interval",error=error,cleanup=cleanup,braking=audit,
                    student_requests=audit["accepted_student_requests"],stop_reason=self.lease.stop_reason)

    def hold_failure(self, reason):
        self.broker.stop(reason)
        stop = next(event for event in self.lease.events if event["kind"]=="stop_request")
        if not self.terminal_path.exists():
            write_json(self.terminal_path,dict(episode_id=self.lease.identity["episode_id"],
                moving_interval_ended=False,stop_reason=self.lease.stop_reason))
        if not stop.get("receipt",{}).get("driver_accepted"):
            raise RuntimeError("Independent stop command was not accepted")
        self.emit("moving_failure_stop",event=stop)
        stationary=0
        deadline=time.perf_counter()+8
        while time.perf_counter()<deadline:
            native=self.reader.receive()
            physics=native.physics
            fields=physics.fields
            if native.builtin_ai_controlled:
                raise ValueError("Native control authority changed during failure stop")
            held=abs(fields["speedKmh"])<.1 and fields["gas"]<.02 and fields["brake"]>=.98
            stationary=stationary+1 if held else 0
            self.emit("moving_failure_stop_native",packet=physics.packet_id,
                prefix=physics.prefix_bytes.hex(),received=physics.received_monotonic_seconds)
            if stationary>=3:return dict(stationary_verified=True,stop_reason=self.lease.stop_reason)
        raise TimeoutError("No advancing stationary feedback after failure stop")

    def close(self):
        self.reader.close()
        self.mailbox.close()
