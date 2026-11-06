from types import SimpleNamespace

import pytest

from flybrain_drive.matched_corner_entry import corner_entry_measurement
from flybrain_drive import moving_stop_calibration


def make():
    session=dict(generation="same",epoch=4,track="reserved")
    p=dict(session,center_native_xyz_m=[100.,0.,200.],forward_xz=[0.,1.],
           speed_range_kmh=[100.,104.],maximum_along_error_m=2.,
           maximum_lateral_error_m=.5,maximum_direction_error_rad=.06)
    row=dict(session,native_ai=False,tyres_out=0,speed_kmh=102.)
    before=dict(row,packet_id=1,original_position_receipt=1.,
                position_native_xyz_m=[100.2,0.,199.])
    after=dict(row,packet_id=2,original_position_receipt=1.03,
               position_native_xyz_m=[100.2,0.,199.85])
    return p,before,after


def test_accepts_actual_matching_pose_and_direction_only():
    p, before, after=make()
    assert corner_entry_measurement(p,before,after,now=1.04)["eligible"]
    before["position_native_xyz_m"]=[100.2,0.,200.7]
    assert not corner_entry_measurement(p,before,after,now=1.04)["eligible"]


def test_rejects_speed_stale_or_authority_mismatch():
    p,before,after=make()
    after["speed_kmh"]=112.
    assert not corner_entry_measurement(p,before,after,now=1.04)["eligible"]
    with pytest.raises(ValueError,match="Fresh"):
        corner_entry_measurement(p,before,after,now=1.1)
    after["native_ai"]=True
    with pytest.raises(ValueError,match="authority"):
        corner_entry_measurement(p,before,after,now=1.04)

def test_advancing_graphics_packets_can_repeat_native_position(monkeypatch):
    plan, before, after = make()
    duplicate = dict(after, packet_id=3, original_position_receipt=1.032)
    expired = dict(duplicate, packet_id=4, original_position_receipt=1.051)
    positions = iter(
        SimpleNamespace(packet_id=row["packet_id"], status="live",
                        received_perf_counter_seconds=row["original_position_receipt"],
                        coordinates_native_xyz_m=row["position_native_xyz_m"],
                        metadata=lambda: {})
        for row in (before, after, duplicate, expired)
    )
    reference = dict(status="completed_diagnostic_teacher_moving_reference",
                     approach=dict(world_position_sdk_xyz_m=[220., 100., 0.],
                                   speed_kmh=130., spline_progress=.35))
    calibration = moving_stop_calibration.MovingStopCalibration(
        reference, SimpleNamespace(snapshot=lambda **_: next(positions)))
    packet = SimpleNamespace(
        raw={"privileged_teacher": {"actor_input": False, "raw": {
            "world_position_x": 200., "world_position_y": 100., "world_position_z": 0.}}},
        received_monotonic_seconds=1., sequence=1)
    sample = SimpleNamespace(fields={"speedKmh": 102.})
    monkeypatch.setattr(moving_stop_calibration, "native_track_fields", lambda _: (0, ()))
    for _ in range(3):
        calibration.observe(packet, sample, .35, lambda *args, **kwargs: None)
    assert calibration.previous_graphics.packet_id == before["packet_id"]
    assert calibration.latest_graphics.packet_id == duplicate["packet_id"]
    assert corner_entry_measurement(plan, before, duplicate, now=1.04)["eligible"]
    calibration.observe(packet, sample, .35, lambda *args, **kwargs: None)
    assert calibration.previous_graphics.packet_id == before["packet_id"]
    with pytest.raises(ValueError, match="Fresh"):
        corner_entry_measurement(plan, before, expired, now=1.055)
