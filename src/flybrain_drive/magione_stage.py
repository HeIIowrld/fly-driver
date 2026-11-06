"""Admission for a measured stationary Magione staging stop; never actor input."""
import base64
import math
from pathlib import Path

from .ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX
from .ac_native_position import decode_diagnostic_position
from .graph import sha256
from .sac_unified_student import read


MAGIONE_MOVING_STAGE_PROFILES = {
    "magione_close_corner_30_v1": "magione_staging_30_v1",
    "magione_alternative_corner_30_v1": "magione_alternative_staging_30_v1",
}


def stop_position(row):
    raw = base64.b64decode(row["prefix_base64"], validate=True)
    sample = decode_diagnostic_position(raw, row["received_perf_counter_seconds"])
    if sample.metadata() != row or row["status"] != "live":
        raise ValueError("Exact original live stop position required")
    return sample


def admit_magione_stage(audit_path, state, physics, pose, *, installed_sha256, now):
    audit_path = Path(audit_path)
    audit = read(audit_path)
    config = read(audit_path.parent/"manifest.json")["config"]
    root = Path(config["args"]["recording"])
    terminal_path = root/"result.json"
    run = read(root/"manifest.json")["config"]
    terminal = read(terminal_path)
    reference_path = Path(run["arguments"]["moving_stop_reference"])
    reference = read(reference_path)
    initial = read(root/"initial-session.json")
    if (audit["status"] != "completed_stop_audit" or audit["moving_stop_calibration_passed"] is not True
            or audit["stationary_verified"] is not True or audit["maximum_tyres_out"] != 0
            or audit["signed_minimum_center_to_border_m"] < 1.8
            or terminal["moving_stop_calibration"]["new_damage"] is not False
            or config["teacher_result_sha256"] != sha256(terminal_path)
            or run["moving_stop_reference_sha256"] != sha256(reference_path)
            or run["geometry_result_sha256"] != config["geometry_result_sha256"]
            or reference["teacher_pacing_profile"] not in MAGIONE_MOVING_STAGE_PROFILES.values()
            or reference["installed_lua_sha256"] != installed_sha256):
        raise ValueError("Exact independently audited same-route Magione staging stop required")

    required = dict(track_id="magione", car_id="bmw_z4_gt3", paused=True, in_menu=False,
                    is_online=False, ai_controlled=False, camera_active=False, expert_probe_active=False)
    if (any(state.get(key) != expected for key, expected in required.items())
            or state["generation"] != initial["generation"] or state["epoch"] != initial["epoch"]
            or state.get("pending_receipt_id") or state.get("error")):
        raise ValueError("Current Magione session must match the paused staging owner")

    start = stop_position(audit["stationary_native_position"])
    if (pose.status not in ("live", "pause") or not 0 <= now-pose.received_perf_counter_seconds <= .08
            or pose.packet_id < start.packet_id
            or math.dist(pose.coordinates_native_xyz_m, start.coordinates_native_xyz_m) > .25):
        raise ValueError("Actual paused position differs from the recorded staging stop")
    fields = physics.fields
    values = iter(PHYSICS_PREFIX.unpack(physics.prefix_bytes))
    native = {name: tuple(next(values) for _ in range(width)) for name, _, width in PHYSICS_LAYOUT}
    if (abs(fields["speedKmh"]) >= .1 or fields["gas"] >= .02 or fields["brake"] <= .98
            or native["gear"] != (2,) or native["autoShifterOn"] != (1,) or native["numberOfTyresOut"] != (0,)):
        raise ValueError("Actual on-track first gear, autoshifter and full-brake standstill required")
    temperatures = list(fields["tyreCoreTemperature"])
    recorded = audit["stationary_native_tyre_core_c"]
    if (len(recorded) != 4 or not all(math.isfinite(t) and abs(t-b)<=3
                                     for t,b in zip(temperatures,recorded,strict=True))):
        raise ValueError("Actual four tyre temperatures differ from the staging stop")
    return dict(schema="flybrain.magione_staged_start.v1", actor_input=False,
                staging_profile=reference["teacher_pacing_profile"],
                generation=state["generation"], epoch=state["epoch"], installed_lua_sha256=installed_sha256,
                geometry_result_sha256=config["geometry_result_sha256"], stage_audit_sha256=sha256(audit_path),
                native_position=pose.metadata(), actual_start_temperature_c=temperatures)
