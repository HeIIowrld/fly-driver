"""Measure the actuator owner's emergency stop on original AC feedback."""
import base64
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, decode_physics_prefix
from .graph import sha256
from .matched_corner_entry import corner_entry_measurement, native_motion_resolved


MONZA_GEOMETRY_PATH = Path("configs/ac-teacher-geometry-result.json")
MONZA_UPSTREAM_STOP_PROFILE = "monza_upstream_teacher_100_v1"
MONZA_MEASURED_STOP_PROFILE = "monza_measured_teacher_101_v1"
MONZA_CHICANE_STOP_PROFILE = "monza_chicane_paced_teacher_101_v1"
MONZA_MEASURED_STOP_PROFILES = (MONZA_MEASURED_STOP_PROFILE, MONZA_CHICANE_STOP_PROFILE)
MONZA_TEACHER_STOP_PROFILES = (MONZA_UPSTREAM_STOP_PROFILE, *MONZA_MEASURED_STOP_PROFILES)

MONZA_UPSTREAM_TARGETS = (20., 40., 60., 80., 102.)
MONZA_UPSTREAM_FINAL_SECONDS = 130
MONZA_UPSTREAM_ENTRY_SPEEDS = (99.5, 100.)
MONZA_UPSTREAM_ENTRY_BOUNDS = (1., .2, .004)



def sdk_to_native_xyz(position):
    # Telemetry publisher emits SDK x=native z, y=native x, z=native y.
    x, y, z = position
    return (y, z, x)


def native_track_fields(sample):
    fields = iter(PHYSICS_PREFIX.unpack(sample.prefix_bytes))
    for name, _, width in PHYSICS_LAYOUT:
        values = tuple(next(fields) for _ in range(width))
        if name == "numberOfTyresOut":
            tyres_out = values[0]
        elif name == "carDamage":
            damage = values
    return tyres_out, damage


def signed_reserve(points, borders):
    distances = []
    memberships = []
    for side in borders:
        a = side
        b = np.roll(side, -1, axis=0)
        delta = b-a
        valid = np.sum(delta**2,axis=1)>1e-12
        a,b,delta = a[valid],b[valid],delta[valid]
        x,y = points[:,None,0],points[:,None,1]
        straddle = (a[:,1]>y)!=(b[:,1]>y)
        crossing = a[:,0]+(y-a[:,1])*delta[:,0]/np.where(delta[:,1]!=0,delta[:,1],1)
        memberships.append(np.count_nonzero(straddle&(x<crossing),axis=1)%2==1)
        relative = points[:,None,:]-a
        t = np.clip(np.sum(relative*delta,axis=2)/np.sum(delta**2,axis=1),0,1)
        distances.append(np.linalg.norm(relative-t[:,:,None]*delta,axis=2).min(1))
    distance = np.minimum(*distances)
    return np.where(memberships[0]^memberships[1],distance,-distance)


def validate_natural_monza_reference(reference):
    """Validate the archived teacher braking witness before a natural-entry stop."""
    if reference.get("pacing_only") is not False:
        return False
    try:
        approach = reference["approach"]
        onset = reference["onset"]
        source = reference["natural_braking_source"]
        events = Path(source["events"])
        diagnostic = json.loads(Path(source["audit"]).read_bytes())
        samples = Path(source["samples"])
        if sha256(samples) != diagnostic["sampled_windows_sha256"] or onset not in json.loads(samples.read_bytes()):
            raise ValueError("Original braking window differs")
        geometry = json.loads(MONZA_GEOMETRY_PATH.read_bytes())["geometry_configuration"]
        delay = onset["sdk_end"] - approach["host_time"]
        if (reference["status"] != "completed_diagnostic_teacher_moving_reference"
                or reference["track"] != "monza" or reference["actor_input"] is not False
                or reference.get("teacher_pacing_profile") is not None
                or reference["approach_phase"] not in ("active_braking", "before_brake")
                or (approach["brake"] > .1) != (reference["approach_phase"] == "active_braking")
                or not (100 <= approach["speed_kmh"] <= 145 and .06 <= approach["spline_progress"] <= .94)
                or not (reference["predicted_minimum_center_clearance_m"] >= 2.7
                        and math.isfinite(reference["predicted_minimum_center_clearance_m"]))
                or not (.12 <= delay <= 1.2 and .25 < onset["request_brake"] <= 1.
                        and 100 <= onset["native_before_kmh"] and .1 < onset["max_native_brake_100ms"] <= 1.)
                or diagnostic["status"] != "completed_teacher_high_speed_diagnostic"
                or sha256(source["audit"]) != source["audit_sha256"]
                or sha256(events) != diagnostic["events_sha256"]
                or reference["route_geometry_sha256"] != geometry["track_sha256"]
                or sha256(geometry["track_csv"]) != geometry["track_sha256"]):
            raise ValueError("Natural braking source or geometry differs")
        found_approach = found_brake = False
        recent_positions = []
        with events.open(encoding="utf-8") as archive:
            for line in archive:
                event = json.loads(line)
                if event["kind"] == "raw_pair":
                    sdk = event["sdk"]
                    payload = base64.b64decode(sdk["wire_base64"], validate=True)
                    if hashlib.sha256(payload).hexdigest() != sdk["wire_sha256"]:
                        raise ValueError("Original SDK wire hash differs")
                    packet = json.loads(payload)["privileged_teacher"]
                    if packet["actor_input"] is not False:
                        raise ValueError("Teacher-only position required")
                    fields = packet["raw"]
                    position = [fields["world_position_" + axis] for axis in "xyz"]
                    if sdk["sequence"] == approach["source_sequence"] and len(recent_positions) == 2:
                        earlier_time, earlier_position = recent_positions[0]
                        motion = [position[i] - earlier_position[i] for i in (1, 0)]
                        length = math.hypot(*motion)
                        forward = reference["forward_native_xz"]
                        found_approach = (sdk["wire_sha256"] == approach["wire_sha256"]
                            and event["host_time"] == approach["host_time"]
                            and event["native"]["fields"]["speedKmh"] == approach["speed_kmh"]
                            and event["native"]["fields"]["brake"] == approach["brake"]
                            and fields["NormalizedSplinePosition"] == approach["spline_progress"]
                            and position == approach["world_position_sdk_xyz_m"]
                            and list(sdk_to_native_xyz(position)) == reference["native_center_xyz_m"]
                            and .03 <= event["host_time"] - earlier_time <= .12
                            and length >= .1 and len(forward) == 2
                            and math.dist(forward, [value / length for value in motion]) <= 1e-9)
                    recent_positions = [*recent_positions[-1:], (event["host_time"], position)]
                elif event["kind"] == "teacher_request":
                    request = event["request"]
                    if request["driver_call_finished_perf_counter_seconds"] == onset["sdk_end"]:
                        found_brake = (request["driver_accepted"] is True
                            and event["automatic_shift_coast"] is False
                            and event["unlimited"] == event["limited"]
                            and request["requested_control"]["brake"] == onset["request_brake"])
                if found_approach and found_brake:
                    break
        if found_approach and found_brake:
            reserve = reference["straight_length_m"]
            if not 0. < reserve <= 500.:
                raise ValueError("Invalid straight stop reserve")
            track = np.genfromtxt(geometry["track_csv"], delimiter=",", names=True)
            borders = [np.column_stack([track[f"{side}_border_x"], track[f"{side}_border_y"]])
                       for side in ("left", "right")]
            p = np.asarray(approach["world_position_sdk_xyz_m"][:2])
            direction = np.asarray(reference["forward_native_xz"])[::-1]
            path = p + np.arange(0, reserve + 1, 1.)[:, None] * direction
            clearance = float(signed_reserve(path, borders).min())
            if not (clearance >= 2.7 and math.isclose(clearance,
                    reference["predicted_minimum_center_clearance_m"], abs_tol=1e-8)):
                raise ValueError("Original mapped stop reserve differs")
            return True
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError("Exact natural Monza braking reference required") from error
    raise ValueError("Original natural Monza braking approach or accepted request missing")


def upstream_teacher_stop_plan(reference):
    """Recompute a continuous geometric lower bound from pinned stop and pose evidence."""
    if (reference.get("track") != "monza" or reference.get("pacing_only") is not True
            or reference.get("actor_input") is not False):
        raise ValueError("Upstream stop is scoped to the Monza teacher")
    proof = Path(reference["pose_audit"])
    stop = Path(reference["stop_distance_audit"])
    if (sha256(proof) != reference["pose_audit_sha256"]
            or sha256(stop) != reference["stop_distance_audit_sha256"]):
        raise ValueError("Upstream original evidence differs")
    prior = json.loads(stop.read_bytes())
    pose_audit = json.loads(proof.read_bytes())
    if (prior.get("stationary_verified") is not True or prior.get("moving_stop_calibration_passed") is not False
            or pose_audit.get("stop_distance_source_sha256") != reference["stop_distance_audit_sha256"]):
        raise ValueError("Original conservative stopping-distance evidence required")
    approach = reference["approach"]
    measured = reference["teacher_pacing_profile"] in MONZA_MEASURED_STOP_PROFILES
    speeds = (100., 101.) if measured else MONZA_UPSTREAM_ENTRY_SPEEDS
    if measured:
        if (pose_audit.get("status") != "completed_measured_pacing_corridor_search"
                or pose_audit.get("speed_range_kmh") != list(speeds)
                or pose_audit.get("bounds") != list(MONZA_UPSTREAM_ENTRY_BOUNDS)):
            raise ValueError("Original measured pacing corridor audit required")
        rows = [r for r in pose_audit["passing"] if r["sequence"] == approach["source_sequence"]]
        if len(rows) != 1 or rows[0]["continuous_lower_m"] < 2.7:
            raise ValueError("One matched measured pose with verified geometry required")
        row = rows[0]
        sequence, progress = row["sequence"], row["progress"]
        xy, tangent = row["position_sdk_xyz_m"][:2], row["forward_sdk_xy"]
        events_pin = pose_audit["teacher_events_sha256"]
    else:
        if pose_audit.get("status") != "completed_discrete_upstream_perturbation_geometry_screen":
            raise ValueError("Original discrete geometry audit required")
        rows = [r for r in pose_audit["results"] if 105 in r["speeds_passing_all_discrete_tests"]]
        if len(rows) != 1:
            raise ValueError("One original upstream pose with discrete geometry evidence required")
        row = rows[0]
        sequence, progress = row["pose"]["original_pose_sequence"], row["pose"]["original_progress"]
        xy, tangent = row["raw_origin_sdk_xy"], row["raw_tangent_sdk_xy"]
        events_pin = pose_audit["events_sha256"]
    point = np.asarray(approach["world_position_sdk_xyz_m"], dtype=float)
    forward = np.asarray(tangent, dtype=float)
    if (point.shape != (3,) or not np.isfinite(point).all() or forward.shape != (2,)
            or not math.isclose(float(np.linalg.norm(forward)), 1., abs_tol=1.e-9)
            or approach["source_sequence"] != sequence or approach["spline_progress"] != progress
            or not np.array_equal(point[:2], xy)):
        raise ValueError("Upstream position or tangent differs from original witness")
    events = Path(reference["source_events"])
    if sha256(events) != events_pin:
        raise ValueError("Original Monza teacher events differ")
    prior_poses = []
    with events.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["kind"] != "raw_pair":
                continue
            sdk = event["sdk"]
            wire = base64.b64decode(sdk["wire_base64"], validate=True)
            if hashlib.sha256(wire).hexdigest() != sdk["wire_sha256"]:
                raise ValueError("Original teacher SDK bytes differ")
            packet = json.loads(wire)["privileged_teacher"]
            original = [packet["raw"]["world_position_" + axis] for axis in "xyz"]
            if sdk["sequence"] == sequence:
                if (packet["actor_input"] is not False or not np.array_equal(original, point)
                        or packet["raw"]["NormalizedSplinePosition"] != progress):
                    raise ValueError("Original upstream pose differs")
                if measured:
                    if len(prior_poses) != 2:
                        raise ValueError("Measured pacing approach needs two preceding original poses")
                    native = event["native"]
                    prefix = base64.b64decode(native["prefix_base64"], validate=True)
                    sample = decode_physics_prefix(prefix, native["received_monotonic_seconds"])
                    motion = point[:2]-np.asarray(prior_poses[0][1][:2])
                    distance = float(np.linalg.norm(motion))
                    if (hashlib.sha256(prefix).hexdigest() != native["prefix_sha256"]
                            or sample.packet_id != native["packet_id"] or native_track_fields(sample)[0] != 0
                            or sdk["wire_sha256"] != row["wire_sha256"] or event["host_time"] != row["host_time"]
                            or sample.fields["speedKmh"] != row["speed_kmh"]
                            or not speeds[0] <= sample.fields["speedKmh"] <= speeds[1]
                            or not .03 <= event["host_time"]-prior_poses[0][0] <= .12
                            or distance < .1 or not np.allclose(motion/distance, forward, atol=1.e-12, rtol=0.)):
                        raise ValueError("Original measured pacing approach or tangent differs")
                break
            prior_poses = [*prior_poses[-1:], (event["host_time"], original)]
        else:
            raise ValueError("Original upstream pose sequence unavailable")
    geometry = json.loads(MONZA_GEOMETRY_PATH.read_bytes())["geometry_configuration"]
    if sha256(geometry["track_csv"]) != geometry["track_sha256"] or pose_audit["map_sha256"] != geometry["track_sha256"]:
        raise ValueError("Pinned Monza route geometry required")
    track = np.genfromtxt(geometry["track_csv"], delimiter=",", names=True)
    borders = [np.column_stack([track[f"{side}_border_x"], track[f"{side}_border_y"]])
               for side in ("left", "right")]
    along, lateral, yaw = MONZA_UPSTREAM_ENTRY_BOUNDS
    upper = speeds[1]
    if not (0 < prior["stop_distance_m"] <= 100 and 100 <= prior["entry_speed_kmh"] <= 145):
        raise ValueError("Finite conservative stop witness required")
    length = prior["stop_distance_m"]*(upper/prior["entry_speed_kmh"])**2 + upper/3.6*.75 + 10.
    steps = np.linspace(-along, length+along, math.ceil((length+2*along)/.1)+1)
    margin = min(float(signed_reserve(point[:2]+chunk[:,None]*forward, borders).min())
                 for chunk in np.array_split(steps, 10))
    # Signed distance to the track border is 1-Lipschitz. Account for half
    # a sampling step, lateral shift, and endpoint drift from heading error.
    bound = margin - .05 - lateral - 2*length*math.sin(yaw/2)
    if not math.isfinite(bound) or bound < 2.7:
        raise ValueError("Continuous Monza upstream stop clearance below 2.7m")
    return dict(center_native_xyz_m=sdk_to_native_xyz(point), forward_xz=forward[::-1].tolist(),
                speed_range_kmh=speeds, maximum_along_error_m=along,
                maximum_lateral_error_m=lateral, maximum_direction_error_rad=yaw,
                predicted_minimum_center_clearance_m=bound, straight_length_m=length)


def upstream_heading_match(plan, previous, position, speed, now):
    """An unavailable or stale original motion witness cannot open the stop gate."""
    if previous is None or not 0. < position.received_perf_counter_seconds-previous.received_perf_counter_seconds <= .05:
        return None
    session = dict(generation="teacher", epoch=0, track="monza")
    # The enclosing live teacher guard owns actual session and authority checks.
    def entry(pos):
        return dict(session, native_ai=False, tyres_out=0, speed_kmh=speed,
                    packet_id=pos.packet_id, original_position_receipt=pos.received_perf_counter_seconds,
                    position_native_xyz_m=pos.coordinates_native_xyz_m)
    if (position.packet_id <= previous.packet_id
            or not 0 <= now-position.received_perf_counter_seconds <= .05
            or not native_motion_resolved(previous.coordinates_native_xyz_m, position.coordinates_native_xyz_m)):
        return None
    return corner_entry_measurement(dict(plan, **session), entry(previous), entry(position), now=now)


class MovingStopCalibration:
    def __init__(self, reference, positions):
        if reference["status"] != "completed_diagnostic_teacher_moving_reference":
            raise ValueError("Exact completed teacher approach reference required")
        approach = reference["approach"]
        self.center = sdk_to_native_xyz(approach["world_position_sdk_xyz_m"])
        self.speed = float(approach["speed_kmh"])
        self.progress = float(approach["spline_progress"])
        profile = reference.get("teacher_pacing_profile")
        self.upstream_plan = None
        self.match_distance_m = {
            "magione_alternative_corner_30_v1": .55,
            "magione_alternative_staging_30_v1": .20,
        }.get(profile, 2.5)
        magione_profiles = {
            "magione_close_corner_60_v1": (56, 4, .314453125, .314453125, 30., 2.),
            "magione_close_corner_30_v1": (27, 3, .314453125, .314453125, 30., 2.),
            "magione_staging_30_v1": (27, 3, .248, .252, 18., 2.2),
            "magione_alternative_corner_30_v1": (27.4, .6, .0744, .0747, 24., 2.8),
            "magione_alternative_staging_30_v1": (28.8, .8, .0050, .0052, 28., 3.),
        }
        if profile in MONZA_TEACHER_STOP_PROFILES:
            self.upstream_plan = upstream_teacher_stop_plan(reference)
            self.speed = self.upstream_plan["speed_range_kmh"][1]
        elif profile in magione_profiles:
            expected_speed, tolerance, lower, upper, reserve, margin = magione_profiles[profile]
            if (reference.get("track") != "magione" or reference.get("pacing_only") is not True
                    or not lower <= self.progress <= upper or self.speed != expected_speed
                    or reference.get("straight_length_m", 0) < reserve
                    or reference.get("predicted_minimum_center_clearance_m", 0) < margin):
                raise ValueError("Reviewed Magione close-corner stop reference required")
            self.speed_tolerance = tolerance
        elif profile == "monza_straight_120_v1":
            if not .028<=self.progress<=.040 or reference.get("expected_calibration_speed_kmh")!=118:
                raise ValueError("Reviewed capped Monza straight and target required")
            self.speed = float(reference["expected_calibration_speed_kmh"])
            lower_progress=.028
        else:
            lower_progress=.06
        natural = profile not in magione_profiles and validate_natural_monza_reference(reference)
        self.natural_brake_active = reference["approach_phase"] == "active_braking" if natural else None
        if self.upstream_plan is not None:
            self.speed_tolerance = 0.
        elif profile not in magione_profiles:
            screened = ((reference.get("pacing_only") is True or natural)
                        and reference.get("predicted_minimum_center_clearance_m", 0) >= 2.7
                        and 100 <= self.speed <= 145 and lower_progress <= self.progress <= .94)
            original = 120 <= self.speed <= 145 and .3 < self.progress < .45
            if not (screened or original):
                raise ValueError("Reviewed Monza intermediate speed calibration required")
            self.speed_tolerance = 4 if screened else 8
        self.positions = positions
        self.entry_plan = None
        self.registration = []
        self.last_graphics = None
        self.latest_graphics = self.previous_graphics = None

    def observe(self, packet, control_native, progress, emit):
        if not self.progress - .035 <= progress <= self.progress + .003:
            return None
        raw = packet.raw["privileged_teacher"]
        if raw["actor_input"] is not False:
            raise ValueError("Privileged position must stay outside the actor")
        sdk_position = sdk_to_native_xyz(tuple(raw["raw"]["world_position_" + k] for k in "xyz"))
        position = self.positions.snapshot(maximum_age=.05)
        receipt = position.received_perf_counter_seconds
        sample_age = receipt - packet.received_monotonic_seconds
        error = math.dist(sdk_position, position.coordinates_native_xyz_m)
        distance = math.dist(self.center, position.coordinates_native_xyz_m)
        speed = control_native.fields["speedKmh"]
        tyres_out, damage = native_track_fields(control_native)
        if position.status != "live" or tyres_out != 0:
            emit("stop_calibration_approach_rejected", position=position.metadata(), progress=progress,
                 sdk_sequence=packet.sequence, native_tyres_out=tyres_out, speed_kmh=speed,
                 physics_packet=control_native.packet_id, physics_prefix=control_native.prefix_bytes.hex(),
                 physics_received=control_native.received_monotonic_seconds, actor_input=False)
            raise ValueError("Live zero-tyres-out approach required")
        if position.packet_id != self.last_graphics:
            self.last_graphics = position.packet_id
            if self.latest_graphics is not None and native_motion_resolved(
                    self.latest_graphics.coordinates_native_xyz_m, position.coordinates_native_xyz_m):
                self.previous_graphics = self.latest_graphics
            self.latest_graphics = position
            self.registration.append(error if 0 <= sample_age <= .05 else math.inf)
        emit("stop_calibration_registration", position=position.metadata(),
             sdk_position_native_xyz_m=sdk_position, sdk_sequence=packet.sequence,
             sdk_received=packet.received_monotonic_seconds, original_position_error_m=error,
             distance_to_gate_m=distance, progress=progress, speed_kmh=speed)
        matched = distance <= self.match_distance_m and abs(speed-self.speed) <= self.speed_tolerance
        if self.entry_plan is not None:
            plan = self.entry_plan
            x, _, z = position.coordinates_native_xyz_m
            cx, _, cz = plan["center_native_xyz_m"]
            tx, tz = plan["forward_xz"]
            along = (x-cx)*tx+(z-cz)*tz
            side = -(x-cx)*tz+(z-cz)*tx
            low, high = plan["speed_range_kmh"]
            matched = (abs(along) <= plan["maximum_along_error_m"]
                       and abs(side) <= plan["maximum_lateral_error_m"] and low <= speed <= high)
        if self.upstream_plan is not None:
            measurement = upstream_heading_match(self.upstream_plan, self.previous_graphics, position,
                                                  speed, time.perf_counter())
            matched = measurement is not None and measurement["eligible"]
            if measurement is not None:
                emit("upstream_teacher_entry_check", position=position.metadata(), measurement=measurement,
                     actor_input=False)
        if matched and self.natural_brake_active is not None:
            brake = control_native.fields["brake"]
            if not 0. <= brake <= 1. or (brake > .1) != self.natural_brake_active:
                emit("stop_calibration_phase_mismatch", sdk_sequence=packet.sequence,
                     original_native_brake=brake, expected_active=self.natural_brake_active)
                matched = False
        if matched:
            recent = self.registration[-12:]
            if len(recent) != 12 or max(recent) > 2.5:
                raise ValueError("Original SDK/native moving position registration not established")
            return position, control_native, damage
        if progress > self.progress + .001:
            raise ValueError("Reference approach passed without a matched native stop start")
        return None

    def stop(self, gate, *, session, reader, guard, emit, clock=time.perf_counter):
        initial_position, initial_physics, damage = gate
        guard()
        if clock()-initial_position.received_perf_counter_seconds > .05:
            raise TimeoutError("Matched original native pose expired before calibration stop")
        request = session.submit(0., 0., 1.)
        emit("stop_calibration_request", request=request, cleanup_not_learned_braking=True,
             initial_position=initial_position.metadata(), speed_kmh=initial_physics.fields["speedKmh"],
             entry_physics_prefix=initial_physics.prefix_bytes.hex(),
             entry_physics_received=initial_physics.received_monotonic_seconds,
             entry_physics_packet=initial_physics.packet_id)
        return self.watch_stop(gate, request, reader=reader, guard=guard, emit=emit, clock=clock)

    def watch_stop(self, gate, request, *, reader, guard, emit, clock=time.perf_counter):
        initial_position, _, damage = gate
        start = request["driver_call_finished_perf_counter_seconds"]
        previous = initial_position.coordinates_native_xyz_m
        last_position = initial_position.packet_id
        path = 0.
        max_tyres = 0
        new_damage = False
        stationary = 0
        while clock()-start < 8.:
            guard()
            sample = reader.receive()
            pos = self.positions.snapshot(maximum_age=.08)
            if pos.status != "live":
                raise ValueError("Lost live graphics during independent stop")
            tyres_out, values = native_track_fields(sample)
            max_tyres = max(tyres_out, max_tyres)
            new_damage |= any(x > base+.01 for x,base in zip(values,damage))
            if pos.packet_id != last_position:
                path += math.dist(previous,pos.coordinates_native_xyz_m)
                previous = pos.coordinates_native_xyz_m
                last_position = pos.packet_id
            fields = sample.fields
            stopped = (abs(fields["speedKmh"]) < .1 and fields["gas"] < .02 and fields["brake"] >= .98)
            stationary = stationary+1 if stopped else 0
            emit("stop_calibration_native", position=pos.metadata(), physics_packet=sample.packet_id,
                 physics_prefix=sample.prefix_bytes.hex(), physics_received=sample.received_monotonic_seconds,
                 speed_kmh=fields["speedKmh"], brake=fields["brake"], tyres_out=tyres_out, path_m=path)
            if stationary >= 3:
                return dict(status="completed_stop_measurement" if not (max_tyres or new_damage)
                            else "stopped_outside_calibration_limits", stationary_verified=True,
                            stop_seconds=clock()-start, path_m=path, maximum_tyres_out=max_tyres,
                            new_damage=new_damage, registration_pairs=len(self.registration),
                            maximum_gate_registration_error_m=max(self.registration[-12:]),
                            student_actuation=False, actor_input=False)
        raise TimeoutError("Independent full brake did not reach verified standstill within 8s")
