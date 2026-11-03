"""Pinned Windows graphics prefix: no native calls or upstream execution."""

import ast
import struct
from pathlib import Path

import pytest

from flybrain_drive.ac_native_clock import GRAPHICS_PREFIX, decode_graphics_timing


def test_graphics_prefix_matches_pinned_windows_layout_stops_before_route_distance():
    path = (Path(__file__).resolve().parents[1] / 'assets/external/ac-native-memory-20261027'
            / 'upstream/sim_info.py')
    tree = ast.parse(path.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SPageFileGraphic')
    fields = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == '_fields_' for t in n.targets))
    offset, offsets = 0, {}
    for item in fields.elts:
        name, kind = item.elts
        count = 1
        if isinstance(kind, ast.BinOp):
            kind, count = kind.left, ast.literal_eval(kind.right)
        name = ast.literal_eval(name)
        offsets[name] = offset
        offset += count * (2 if kind.id == 'c_wchar' else 4)
        if name == 'sessionTimeLeft':
            break
    assert offset == GRAPHICS_PREFIX.size == 156
    assert offsets['iCurrentTime'] == 140 and offsets['completedLaps'] == 132
    assert 'distanceTraveled' not in offsets


def test_graphics_timing_keeps_raw_units_and_repeated_times_without_fake_clock():
    raw = GRAPHICS_PREFIX.pack(10, 2, 0, bytes(120), 0, 1, 123, 0, 0, -1.)
    sample = decode_graphics_timing(raw, 20.)
    assert sample.status == 'live' and sample.current_time_raw == 123
    assert sample.prefix_bytes == raw and not hasattr(sample, 'simulation_seconds')
    assert not hasattr(sample, 'position')
    changed = bytearray(raw)
    struct.pack_into('<i', changed, 4, 4)
    with pytest.raises(ValueError, match='ABI'):
        decode_graphics_timing(bytes(changed), 20.)
    with pytest.raises(ValueError, match='Exact'):
        decode_graphics_timing(raw + bytes(4), 20.)


def test_native_reader_only_opens_timing_prefix_and_retains_stale_snapshot(monkeypatch):
    import flybrain_drive.ac_native_clock as native

    class Mapping:
        closed = False

        def read(self):
            return GRAPHICS_PREFIX.pack(10, 2, 0, bytes(120), 0, 1, 123, 0, 0, -1.)

        def close(self):
            self.closed = True

    mapping, calls = Mapping(), []

    def open_existing(name, size):
        calls.append((name, size))
        return mapping

    monkeypatch.setattr(native, 'ExistingNamedMapping', open_existing)
    with native.NativeGraphicsTimingReader() as reader:
        a, b = reader.snapshot(), reader.snapshot()
        assert a.packet_id == b.packet_id == 10
        assert a.current_time_raw == b.current_time_raw == 123
    assert mapping.closed and calls == [('Local\\acpmf_graphics', 156)]
