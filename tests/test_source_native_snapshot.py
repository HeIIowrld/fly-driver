import base64
import json
import struct
import ctypes
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.ac_transport import decode_telemetry
from flybrain_drive.source_native_snapshot import decode_source_native
from test_ac_tcp import load_app


def packet(**changes):
    payload = bytearray(PHYSICS_PREFIX.size)
    struct.pack_into('<i',payload,0,42)
    row=dict(protocol='flybrain.ac.source_native.v1',repeated_copy_equal=True,
        physical_tick_time_verified=False,camera_exposure_sync_verified=False,
        qpc_started_ticks=100000,qpc_finished_ticks=100100,qpc_frequency_hz=1000000,
        prefix_base64=base64.b64encode(payload).decode())
    row.update(changes)
    return decode_telemetry(json.dumps(dict(steps=1,timestamp_ac=2,
        source_native_snapshot=row)).encode(),30.)


def test_original_source_bundle_receipt_and_bytes_are_preserved():
    value=decode_source_native(packet(),SimpleNamespace(frequency=1000000,ticks=lambda:120000))
    assert value.packet_id==42 and value.received_monotonic_seconds==30.
    assert value.source_time_seconds is None and not value.simulation_clock_verified


def test_late_ipc_parse_cannot_re_date_an_old_capture():
    with pytest.raises(TimeoutError,match='source QPC'):
        decode_source_native(packet(),SimpleNamespace(frequency=1000000,ticks=lambda:151000))


@pytest.mark.parametrize('changes',[dict(qpc_frequency_hz=2),dict(qpc_started_ticks=101000),
    dict(qpc_finished_ticks=110000),dict(physical_tick_time_verified=True),
    dict(camera_exposure_sync_verified=True),dict(prefix_base64='invalid!')])
def test_clock_and_unpromoted_physical_contract_fail_closed(changes):
    with pytest.raises((ValueError,TypeError)):
        decode_source_native(packet(**changes),SimpleNamespace(frequency=1000000,ticks=lambda:120000))


def test_publisher_retries_a_changed_native_prefix_without_mixing_it(monkeypatch):
    app=load_app()
    witness=object.__new__(app.SourceNativeSnapshot)
    first,second=bytes(308),b'\x01'+bytes(307)
    copies=iter([first,second,second,second])
    witness.ctypes=SimpleNamespace(string_at=lambda *args:next(copies))
    witness.base64=base64
    witness.view=1
    witness.frequency=1000000
    ticks=iter([100000,100100,101000,101100])
    witness.ticks=lambda:next(ticks)
    calls=[]
    monkeypatch.setattr(app,'read_body',lambda *args:dict(sdk_attempt=len(calls)))
    def read_teacher(*args):
        calls.append(1)
        return dict(attempt=len(calls))
    monkeypatch.setattr(app,'read_privileged_teacher',read_teacher)
    row=witness.capture(None,None)
    assert row['sdk_attempt']==1 and row['privileged_teacher']['attempt']==2
    assert row['source_native_snapshot']['attempts']==2
    assert base64.b64decode(row['source_native_snapshot']['prefix_base64'])==second
