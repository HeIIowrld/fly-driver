"""Recompute the owner stop and SDK/native registration from immutable bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_native_position import decode_diagnostic_position
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.moving_stop_calibration import native_track_fields, signed_reserve
from flybrain_drive.sac_unified_student import read


def position(row):
    raw = base64.b64decode(row["prefix_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"]:
        raise ValueError("Original graphics bytes differ")
    sample = decode_diagnostic_position(raw, row["received_perf_counter_seconds"])
    if sample.packet_id != row["packet_id"] or sample.status != row["status"]:
        raise ValueError("Original graphics packet differs")
    return sample


def clearance(points, geometry, *, signed=False):
    config = read(geometry)["geometry_configuration"]
    path = config["track_csv"]
    if sha256(path) != config["track_sha256"]:
        raise ValueError("Original privileged track geometry differs")
    track = np.genfromtxt(path, delimiter=",", names=True)
    sides = [np.column_stack((track[f"{side}_border_x"],track[f"{side}_border_y"]))
             for side in ("left","right")]
    if signed:
        return signed_reserve(np.asarray(points)[:, [2, 0]], sides)
    a = np.concatenate([x[:-1] for x in sides])
    delta = np.concatenate([np.diff(x,axis=0) for x in sides])
    valid = np.sum(delta**2,axis=1)>0
    a, delta = a[valid], delta[valid]
    norm = np.sum(delta**2,axis=1)
    out = []
    for native in points:
        # Same ordering as the teacher's horizontal coordinates.
        point = np.array([native[2],native[0]])
        t = np.clip(np.sum((point-a)*delta,axis=1)/norm,0,1)
        out.append(float(np.linalg.norm(a+t[:,None]*delta-point,axis=1).min()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--magione-geometry",
        help="Exact privileged geometry for the separate direct Magione teacher run")
    args = parser.parse_args()
    root = Path(args.recording)
    teacher = root if args.magione_geometry else root/"teacher"
    geometry = args.magione_geometry or "configs/ac-teacher-geometry-result.json"
    terminal = read(teacher/"result.json")
    if args.magione_geometry:
        config = read(teacher/"manifest.json")["config"]
        if (config.get("expected_track") != "magione"
                or config.get("geometry_result_sha256") != sha256(geometry)):
            raise ValueError("Magione stop and geometry provenance differ")
    out = begin_run(args.output,dict(args=vars(args),program_sha256=sha256(__file__),
        teacher_result_sha256=sha256(teacher/"result.json"),geometry_result_sha256=sha256(geometry),
        seed=None,actor_input=False))
    result = dict(status="failed",teacher_status=terminal["status"],teacher_reason=terminal.get("reason"),
                  student_requests=0,actor_input=False)
    try:
        path = teacher/"events.jsonl"
        if sha256(path) != terminal["events_sha256"]:
            raise ValueError("Original owner event log differs")
        registrations = {}
        registered_poses = []
        requests = []
        samples = []
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            kind = row["kind"]
            if kind == "stop_calibration_registration":
                pos = position(row["position"])
                error = np.linalg.norm(np.asarray(pos.coordinates_native_xyz_m)-row["sdk_position_native_xyz_m"])
                np.testing.assert_allclose(error,row["original_position_error_m"],atol=1e-9,rtol=0)
                if pos.packet_id not in registrations:
                    registered_poses.append(pos)
                registrations.setdefault(pos.packet_id,dict(error_m=float(error), speed_kmh=row["speed_kmh"],
                    receipt_skew_seconds=pos.received_perf_counter_seconds-row["sdk_received"]))
            elif kind == "stop_calibration_request":
                requests.append(row)
            elif kind == "stop_calibration_native":
                pos = position(row["position"])
                sample = decode_physics_prefix(bytes.fromhex(row["physics_prefix"]),row["physics_received"])
                tyres,damage = native_track_fields(sample)
                if (sample.packet_id!=row["physics_packet"] or tyres!=row["tyres_out"]
                        or sample.fields["speedKmh"]!=row["speed_kmh"] or sample.fields["brake"]!=row["brake"]):
                    raise ValueError("Original stop physics differs")
                samples.append((pos,sample,tyres,damage))
        result["registration_packets"] = len(registrations)
        if registrations:
            result["registration_error_m_max"] = max(v["error_m"] for v in registrations.values())
            result["max_approach_speed_kmh"] = max(v["speed_kmh"] for v in registrations.values())
        result["stop_feedback_packets"] = len(samples)
        if args.magione_geometry:
            result["heldout_failure_stop"] = terminal.get("heldout_failure_stop")
        if requests:
            if len(requests)!=1 or requests[0]["request"]["requested_control"]!=dict(steering=0.,throttle=0.,brake=1.):
                raise ValueError("One explicit independent stop request required")
            start = position(requests[0]["initial_position"])
            if "entry_physics_prefix" in requests[0]:
                entry = decode_physics_prefix(bytes.fromhex(requests[0]["entry_physics_prefix"]),
                                              requests[0]["entry_physics_received"])
                if (entry.packet_id != requests[0]["entry_physics_packet"]
                        or entry.fields["speedKmh"] != requests[0]["speed_kmh"]):
                    raise ValueError("Original entry physics differs")
                result["entry_native_brake"] = entry.fields["brake"]
            poses = {p.packet_id:p.coordinates_native_xyz_m for p,_,_,_ in samples}
            points = np.asarray([start.coordinates_native_xyz_m,*[
                v for k,v in poses.items() if k>start.packet_id]])
            distances = clearance(points, geometry)
            physics = [s for _,s,_,_ in samples]
            stationary = len(physics)>=3 and all(abs(v.fields["speedKmh"])<.1 and v.fields["brake"]>=.98
                                                 for v in physics[-3:])
            result.update(entry_speed_kmh=requests[0]["speed_kmh"],stop_distance_m=float(
                np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                maximum_tyres_out=max((x[2] for x in samples),default=None),
                stationary_verified=stationary,unsigned_minimum_center_to_border_m=min(distances))
            signed_margin = clearance(points, geometry, signed=True)
            result["signed_minimum_center_to_border_m"] = float(signed_margin.min())
            heading = None
            center = np.asarray(start.coordinates_native_xyz_m)[[0,2]]
            for earlier in reversed(registered_poses):
                if not 0 < start.received_perf_counter_seconds-earlier.received_perf_counter_seconds <= .12:
                    continue
                motion = center-np.asarray(earlier.coordinates_native_xyz_m)[[0,2]]
                distance = np.linalg.norm(motion)
                if distance>=.1:
                    heading = (motion/distance).tolist()
                    break
            result.update(entry_native_position=start.metadata(), entry_forward_native_xz=heading)
            if args.magione_geometry:
                if heading is None:
                    raise ValueError("Original moving heading unavailable at Magione stop request")
                result.update(stationary_native_position=samples[-1][0].metadata(),
                    stationary_native_physics_packet=physics[-1].packet_id,
                    stationary_native_physics_prefix_sha256=hashlib.sha256(physics[-1].prefix_bytes).hexdigest(),
                    stationary_native_tyre_core_c=list(physics[-1].fields["tyreCoreTemperature"]))
            np.savez_compressed(out/"diagnostic-stop-path.npz",native_xyz_m=points,unsigned_border_distance_m=distances)
        result["status"] = "completed_stop_audit" if requests else "completed_approach_failure_audit"
        result["moving_stop_calibration_passed"] = (
            terminal["status"]=="teacher_moving_stop_calibration_completed"
            and result.get("stationary_verified") is True and result.get("maximum_tyres_out")==0
            and (not args.magione_geometry or result["signed_minimum_center_to_border_m"]>=1.8))
    except Exception as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
