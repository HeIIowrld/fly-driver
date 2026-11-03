"""Pinned Windows ABI and explicit unknown clocks for read-only native AC physics."""

import ast
import struct
from pathlib import Path

import pytest

from flybrain_drive.ac_native_physics import (
    BODY_FIELDS,
    PHYSICS_LAYOUT,
    PHYSICS_PREFIX,
    decode_physics_prefix,
    decode_static_identity_prefix,
    repeated_snapshot,
)


def test_layout_exactly_matches_reviewed_upstream_prefix_without_executing_it():
    path = (Path(__file__).resolve().parents[1] / "assets/external/ac-native-memory-20261027"
            / "upstream/sim_info.py")
    tree = ast.parse(path.read_text(encoding="utf8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SPageFilePhysics")
    fields = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_fields_" for t in n.targets))
    prefix = []
    for item in fields.elts:
        name, kind = item.elts
        count = 1
        if isinstance(kind, ast.BinOp) and isinstance(kind.op, ast.Mult):
            kind, count = kind.left, ast.literal_eval(kind.right)
        prefix.append((ast.literal_eval(name), {"c_int32": "i", "c_float": "f"}[kind.id], count))
        if prefix[-1][0] == "localAngularVel":
            break
    assert tuple(prefix) == PHYSICS_LAYOUT and PHYSICS_PREFIX.size == 308


def test_decoder_preserves_four_wheel_values_excludes_pose_and_invents_no_clock():
    data = bytearray(308)
    struct.pack_into("<i", data, 0, 101)
    struct.pack_into("<4f", data, 72, 1000, 1100, 2000, 2100)
    struct.pack_into("<4f", data, 152, 70, 71, 72, 73)
    struct.pack_into("<3f", data, 296, .1, .2, .3)
    sample = decode_physics_prefix(bytes(data), 10.)
    assert sample.packet_id == 101
    assert sample.fields["wheelLoad"] == (1000, 1100, 2000, 2100)
    assert sample.fields["tyreCoreTemperature"] == (70, 71, 72, 73)
    assert set(sample.fields) == BODY_FIELDS
    assert sample.source_time_seconds is None and not sample.simulation_clock_verified
    assert not sample.atomicity_verified
    assert not {"heading", "velocity", "pitch", "roll"} & sample.fields.keys()
    with pytest.raises(TypeError):
        sample.fields["heading"] = 0
    struct.pack_into("<f", data, 72, float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        decode_physics_prefix(bytes(data), 10.)


def test_repeat_copy_guard_rejects_torn_samples():
    class Changing:
        counter = 0

        def read(self):
            self.counter += 1
            return struct.pack("<i", self.counter) + bytes(304)

    with pytest.raises(RuntimeError, match="changed"):
        repeated_snapshot(Changing(), attempts=3)


def test_static_identity_uses_windows_utf16_and_stops_before_personal_fields():
    data = bytearray(200)
    for text, start in (("1.7", 0), ("1.16.4", 30), ("bmw_z4_gt3", 68), ("monza", 134)):
        encoded = text.encode("utf-16-le")
        data[start:start+len(encoded)] = encoded
    identity = decode_static_identity_prefix(bytes(data))
    assert identity["car_model"] == "bmw_z4_gt3" and identity["track"] == "monza"
    assert not any("player" in key for key in identity)
    with pytest.raises(ValueError, match="200-byte"):
        decode_static_identity_prefix(bytes(data) + b"private")


def test_reader_opens_only_existing_prefix_and_rejects_stale_or_reset(monkeypatch):
    import flybrain_drive.ac_native_physics as native

    class Mapping:
        frame = struct.pack("<i", 10) + bytes(304)
        closed = False

        def read(self):
            return self.frame

        def close(self):
            self.closed = True

    mapping, calls = Mapping(), []

    def open_existing(name, size):
        calls.append((name, size))
        return mapping

    monkeypatch.setattr(native, "ExistingNamedMapping", open_existing)
    with native.NativePhysicsReader(timeout=.01, poll_seconds=.001) as reader:
        assert reader.receive().packet_id == 10
        with pytest.raises(TimeoutError, match="no new"):
            reader.receive()
        mapping.frame = struct.pack("<i", 12) + bytes(304)
        assert reader.receive().packet_id == 12
        mapping.frame = struct.pack("<i", 1) + bytes(304)
        with pytest.raises(ValueError, match="sequence reset"):
            reader.receive()
    assert calls == [("Local\\acpmf_physics", 308)] and mapping.closed


def test_snapshot_can_use_explicit_high_resolution_receipt_clock():
    class Mapping:
        def read(self):
            return struct.pack('<i', 10)+bytes(304)

    sample = repeated_snapshot(Mapping(), clock=lambda: 123.456789)
    assert sample.received_monotonic_seconds == 123.456789
