import ast
import ctypes
from pathlib import Path

import pytest

from flybrain_drive.ac_native_position import (
    POSITION_PREFIX,
    NativeDiagnosticPositionReader,
    decode_diagnostic_position,
)


def prefix(packet=1, xyz=(100., 2., -33.)):
    raw = bytearray(POSITION_PREFIX.pack(packet, *xyz))
    raw[4:8] = (2).to_bytes(4, 'little')
    raw[8:12] = (3).to_bytes(4, 'little')
    return bytes(raw)


def test_upstream_ast_layout_offsets_are_windows_utf16_pack4_without_importing_it():
    tree = ast.parse(Path('assets/external/ac-native-memory-20261027/upstream/sim_info.py')
                     .read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SPageFileGraphic')
    fields = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                  and n.targets[0].id == '_fields_')
    types = dict(c_int32=ctypes.c_int32, c_float=ctypes.c_float,
                 c_wchar=ctypes.c_uint16, AC_STATUS=ctypes.c_int32,
                 AC_SESSION_TYPE=ctypes.c_int32, AC_FLAG_TYPE=ctypes.c_int32)
    def evaluate(node):
        if isinstance(node, ast.Name):
            return types[node.id]
        assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
        return evaluate(node.left) * ast.literal_eval(node.right)
    layout = type('VerifiedWindowsGraphics', (ctypes.Structure,), dict(_pack_=4,
        _fields_=[(ast.literal_eval(n.elts[0]), evaluate(n.elts[1])) for n in fields.elts]))
    assert layout.carCoordinates.offset == 252
    assert POSITION_PREFIX.size == layout.carCoordinates.offset + 12
    value = layout()
    value.packetId, value.status, value.session = 4, 2, 3
    value.carCoordinates[:] = [100., 2., -33.]
    decoded = decode_diagnostic_position(bytes(value)[:264], 123.)
    assert decoded.coordinates_native_xyz_m == (100., 2., -33.)
    assert decoded.status == 'live' and decoded.metadata()['actor_input'] is False


def test_repeated_snapshot_retains_receipt_and_stale_or_mutated_packet_is_rejected():
    reader = object.__new__(NativeDiagnosticPositionReader)
    now, raw = [10.], [prefix()]
    reader.clock, reader.last = lambda: now[0], None
    reader.mapping = type('Mapping', (), {'read': lambda _: raw[0]})()
    first = reader.snapshot()
    now[0] = 10.04
    assert reader.snapshot() is first
    now[0] = 10.081
    with pytest.raises(TimeoutError):
        reader.snapshot()
    raw[0] = prefix(xyz=(101., 2., -33.))
    with pytest.raises(ValueError, match='different bytes'):
        reader.snapshot()
    raw[0] = prefix(packet=2)
    assert reader.snapshot().received_perf_counter_seconds == now[0]
    raw[0] = prefix(packet=1)
    with pytest.raises(ValueError, match='reset'):
        reader.snapshot()


def test_nonfinite_position_cannot_be_used_as_a_diagnostic_bound():
    with pytest.raises(ValueError):
        decode_diagnostic_position(prefix(xyz=(float('nan'), 0., 0.)), 1.)
