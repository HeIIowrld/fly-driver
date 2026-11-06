import json
import struct

import pytest

from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.ac_native_position import POSITION_PREFIX, decode_diagnostic_position
from flybrain_drive.graph import sha256
from flybrain_drive.magione_stage import admit_magione_stage
from flybrain_drive.moving_stop_calibration import MovingStopCalibration
from flybrain_drive import moving_flight_trial


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def position(packet, status, point, receipt):
    raw = bytearray(POSITION_PREFIX.pack(packet, *point))
    struct.pack_into("<i", raw, 4, status)
    return decode_diagnostic_position(bytes(raw), receipt)


def test_alternative_corner_stop_requires_its_reviewed_envelope():
    reference = dict(
        status="completed_diagnostic_teacher_moving_reference",
        track="magione", teacher_pacing_profile="magione_alternative_corner_30_v1",
        pacing_only=True, straight_length_m=25.02,
        predicted_minimum_center_clearance_m=2.89,
        approach=dict(world_position_sdk_xyz_m=(-260., -161., 5.4),
                      speed_kmh=27.4, spline_progress=.07455),
    )
    calibration = MovingStopCalibration(reference, None)
    assert calibration.match_distance_m == .55
    assert calibration.speed_tolerance == .6
    for change in (dict(track="monza"), dict(straight_length_m=23.99),
                   dict(predicted_minimum_center_clearance_m=2.79),
                   dict(approach={**reference["approach"], "spline_progress": .07471})):
        with pytest.raises(ValueError, match="Reviewed Magione"):
            MovingStopCalibration({**reference, **change}, None)


def test_alternative_corner_transfer_leaves_a_measured_speed_reserve(tmp_path, monkeypatch):
    reference = dict(teacher_pacing_profile="magione_alternative_corner_30_v1",
                     straight_length_m=25.02)
    audit = dict(entry_speed_kmh=27.45, stop_distance_m=2.45,
                 entry_native_position=dict(coordinates_native_xyz_m=(1., 2., 3.)),
                 entry_forward_native_xz=(1., 0.))
    monkeypatch.setattr(moving_flight_trial, "admit_calibration", lambda *args: (reference, audit))
    monkeypatch.setattr(moving_flight_trial, "sha256", lambda path: "digest")
    monkeypatch.setattr(moving_flight_trial, "NativeTeacherPhysicsReader", lambda **kwargs: None)
    monkeypatch.setattr(moving_flight_trial, "FlightDecisionMailbox", lambda name: None)
    from types import SimpleNamespace
    calibration = SimpleNamespace(entry_plan=None)
    identity = dict(generation="g", episode_id="episode", run_id="run", checkpoint_sha256="a"*64)
    plan = dict(schema="flybrain.moving_student_trial.v1", identity=identity,
                reference_sha256="digest", audit_sha256="digest", mailbox="test",
                terminal_path=tmp_path/"terminal.json")
    trial = moving_flight_trial.MovingStudentTrial(
        plan=plan, reference_path="reference", audit_path="audit",
        generation="g", epoch=0, request_id="episode", calibration=calibration,
        session=object(), guard=lambda: None, emit=lambda *args: None,
    )
    assert calibration.entry_plan["speed_range_kmh"] == pytest.approx([21.5, 22.8])
    assert trial.maximum_calibrated_speed_kmh == 27.45


def test_stage_requires_same_stopped_native_pose_and_session(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    folder = root / "audit"
    folder.mkdir()
    ref = write(root / "reference.json", dict(
        teacher_pacing_profile="magione_staging_30_v1", installed_lua_sha256="lua"))
    terminal = write(root / "result.json", dict(moving_stop_calibration=dict(new_damage=False)))
    write(root / "manifest.json", dict(config=dict(arguments=dict(moving_stop_reference=str(ref)),
          moving_stop_reference_sha256=sha256(ref), geometry_result_sha256="geometry")))
    write(root / "initial-session.json", dict(generation="same", epoch=1))
    write(folder / "manifest.json", dict(config=dict(args=dict(recording=str(root)),
          teacher_result_sha256=sha256(terminal), geometry_result_sha256="geometry")))
    staged = position(11, 2, [1., 2., 3.], 2.)
    proof = write(folder / "result.json", dict(status="completed_stop_audit", moving_stop_calibration_passed=True,
        stationary_verified=True, maximum_tyres_out=0, signed_minimum_center_to_border_m=2.4,
        stationary_native_position=staged.metadata(), stationary_native_tyre_core_c=[36.]*4))
    state = dict(track_id="magione",car_id="bmw_z4_gt3",generation="same",epoch=1,
        paused=True,in_menu=False,is_online=False,ai_controlled=False,camera_active=False,expert_probe_active=False)
    values = dict(gear=(2,),autoShifterOn=(1,),brake=(1.,),tyreCoreTemperature=(36.,)*4)
    raw = PHYSICS_PREFIX.pack(*(v for key, _, n in PHYSICS_LAYOUT for v in values.get(key,(0,)*n)))
    sample = decode_physics_prefix(raw, 10.)
    paused = position(12, 3, [1.04, 2., 3.], 10.)
    result = admit_magione_stage(proof,state,sample,paused,installed_sha256="lua",now=10.01)
    assert result["actor_input"] is False
    with pytest.raises(ValueError,match="paused staging owner"):
        admit_magione_stage(proof,dict(state,epoch=2),sample,paused,installed_sha256="lua",now=10.01)
    moved = position(13, 3, [1.4, 2., 3.], 10.)
    with pytest.raises(ValueError,match="position differs"):
        admit_magione_stage(proof,state,sample,moved,installed_sha256="lua",now=10.01)
