"""Natural Monza references require both archived braking and a matched signed stop."""
import base64
import hashlib
import json

import pytest

from flybrain_drive import moving_flight_trial, moving_stop_calibration
from flybrain_drive.graph import sha256
from flybrain_drive.moving_flight_trial import admit_calibration
from flybrain_drive.moving_stop_calibration import MovingStopCalibration


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def reference_and_audit(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    records = []
    for sequence, stamp, x in ((1, 1., 0.), (2, 1.04, 1.), (3, 1.08, 2.)):
        wire = json.dumps({"privileged_teacher": {"actor_input": False, "raw": {
            "world_position_x": x, "world_position_y": 0., "world_position_z": 0.,
            "NormalizedSplinePosition": .15}}}).encode()
        digest = hashlib.sha256(wire).hexdigest()
        records.append(dict(kind="raw_pair",host_time=stamp,
            sdk=dict(sequence=sequence,wire_sha256=digest,wire_base64=base64.b64encode(wire).decode()),
            native=dict(fields=dict(speedKmh=112.,brake=0.))))
    records.append(dict(kind="teacher_request",unlimited=[0.,0.,.5],limited=[0.,0.,.5],
        automatic_shift_coast=False,request=dict(driver_call_finished_perf_counter_seconds=1.4,
            driver_accepted=True,requested_control=dict(brake=.5))))
    events.write_text("\n".join(map(json.dumps,records)),encoding="utf-8")
    onset=dict(sdk_end=1.4,request_brake=.5,native_before_kmh=112.,max_native_brake_100ms=.5)
    samples=write(tmp_path/"samples.json",[onset])
    source=write(tmp_path/"braking-audit.json",dict(status="completed_teacher_high_speed_diagnostic",
        events_sha256=sha256(events),sampled_windows_sha256=sha256(samples)))
    track = tmp_path / "synthetic-test-track.csv"
    track.write_text("left_border_x,left_border_y,right_border_x,right_border_y\n"
        "-10,-5,-4,5\n120,-5,114,5\n120,15,114,10\n-10,15,-4,10\n", encoding="utf-8")
    geometry_sha=sha256(track)
    geometry=write(tmp_path/"geometry.json",
        dict(geometry_configuration=dict(track_csv=str(track),track_sha256=geometry_sha)))
    monkeypatch.setattr(moving_stop_calibration,"MONZA_GEOMETRY_PATH",geometry)
    monkeypatch.setattr(moving_flight_trial,"MONZA_GEOMETRY_PATH",geometry)
    ref=dict(status="completed_diagnostic_teacher_moving_reference",pacing_only=False,track="monza",
        actor_input=False,approach_phase="before_brake",route_geometry_sha256=geometry_sha,
        approach=dict(world_position_sdk_xyz_m=[2.,0.,0.],speed_kmh=112.,spline_progress=.15,
                      source_sequence=3,wire_sha256=digest,host_time=1.08,brake=0.),
        onset=onset,native_center_xyz_m=[0.,0.,2.],forward_native_xz=[0.,1.],
        predicted_minimum_center_clearance_m=5.,straight_length_m=90.,
        natural_braking_source=dict(events=str(events),audit=str(source),audit_sha256=sha256(source),samples=str(samples)))
    ref_path=write(tmp_path/"reference.json",ref)
    recording=tmp_path/"recording"
    teacher=recording/"teacher"
    teacher.mkdir(parents=True)
    write(teacher/"manifest.json",dict(config=dict(moving_stop_reference_sha256=sha256(ref_path))))
    terminal=write(teacher/"result.json",dict(moving_stop_calibration=dict(new_damage=False)))
    auditor=tmp_path/"stop-audit"
    auditor.mkdir()
    write(auditor/"manifest.json",dict(config=dict(args=dict(recording=str(recording)),
        teacher_result_sha256=sha256(terminal),geometry_result_sha256=sha256(geometry))))
    audit=dict(status="completed_stop_audit",moving_stop_calibration_passed=True,stationary_verified=True,
        maximum_tyres_out=0,entry_speed_kmh=112.,stop_distance_m=40.,unsigned_minimum_center_to_border_m=2.2,
        signed_minimum_center_to_border_m=2.2,entry_native_position=dict(coordinates_native_xyz_m=[0.,0.,2.]),
        entry_forward_native_xz=[0.,1.],entry_native_brake=0.)
    return ref_path,ref,write(auditor/"result.json",audit),audit,events


def test_natural_reference_reaches_both_stop_paths(reference_and_audit):
    ref_path,ref,audit_path,audit,_=reference_and_audit
    assert MovingStopCalibration(ref,None).speed_tolerance==4
    assert admit_calibration(ref_path,audit_path)==(ref,audit)


@pytest.mark.parametrize("change",[
    {"signed_minimum_center_to_border_m": -2.2},
    {"entry_forward_native_xz": [.1,.994987437]},
    {"entry_native_position": {"coordinates_native_xyz_m": [0.,0.,5.]}},
    {"entry_native_position": {"coordinates_native_xyz_m": [.56,0.,2.]}},
    {"entry_speed_kmh": 120.},
    {"entry_native_brake": .5},
    {"stop_distance_m": 80.},
])
def test_natural_stop_rejects_unsigned_or_mismatched_evidence(reference_and_audit,change):
    ref_path,_,audit_path,audit,_=reference_and_audit
    write(audit_path,{**audit,**change})
    with pytest.raises(ValueError):
        admit_calibration(ref_path,audit_path)


def test_natural_reference_rejects_changed_source_or_heading(reference_and_audit):
    ref_path,ref,audit_path,_,events=reference_and_audit
    with pytest.raises(ValueError):
        MovingStopCalibration({**ref,"forward_native_xz":[1.,0.]},None)
    with pytest.raises(ValueError):
        MovingStopCalibration({**ref,"straight_length_m":125.},None)
    with events.open("a",encoding="utf-8") as file:
        file.write("\n")
    with pytest.raises(ValueError,match="Exact natural Monza"):
        admit_calibration(ref_path,audit_path)



@pytest.mark.parametrize("profile", moving_stop_calibration.MONZA_TEACHER_STOP_PROFILES)
def test_upstream_teacher_profile_cannot_authorize_student(reference_and_audit,profile):
    ref_path, ref, audit_path, _, _ = reference_and_audit
    write(ref_path, {**ref, "pacing_only": True,
        "teacher_pacing_profile": profile})
    with pytest.raises(ValueError, match="cannot authorize student"):
        admit_calibration(ref_path, audit_path)


def test_chicane_pacing_keeps_the_measured_stop_target_and_earlier_stage():
    from flybrain_drive.teacher_curriculum import monza_chicane_teacher_target
    pace = monza_chicane_teacher_target
    assert pace(102., .08999) == 102.
    assert pace(102., .09) == 80.
    assert pace(102., .119338) == 80.
    assert pace(102., .1203) == 102.
    assert pace(60., .112) == 60.
    with pytest.raises(ValueError, match="progress"):
        pace(102., float("nan"))
