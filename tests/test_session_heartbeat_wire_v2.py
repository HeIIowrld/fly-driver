"""CPU serialization counterexample from1788; no live files/maps/worker/control."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_session_heartbeat_lua_candidate import (
    no_fatal_signal_for_handled_luajit_errors,  # noqa: F401
    python_value,
)
from test_session_heartbeat_service import command, create, fixture

from flybrain_drive.session_heartbeat_candidate import decode_slot
from flybrain_drive.session_heartbeat_terminal_v2 import observe_sync_return

ROOT = Path('integrations/ac_csp/candidates')
WIRE = 38.6236105
RAW = math.nextafter(WIRE, math.inf)


def rounded_service(*, old_service=False):
    lua, _, service = fixture()

    def rounded(value):
        if isinstance(value, float):
            return round(value, 7)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        return value

    lua.globals()[b'json_python'] = lambda value: json.dumps(
        rounded(python_value(value)), separators=(',', ':')).encode()
    proto_file = 'SessionHeartbeatMMF.lua' if old_service else 'SessionHeartbeatMMF-wire-v2.lua'
    proto = lua.execute((ROOT/proto_file).read_bytes())
    lua.globals()[b'proto'] = proto
    if not old_service:
        service = lua.execute((ROOT/'SessionHeartbeatService-wire-v2.lua').read_bytes())
    lua.globals()[b'now'] = RAW
    lua.globals()[b'snapshot'][b'script_clock'] = RAW
    lua.globals()[b'plan'][b'deadline_source_clock'] = 98.
    lua.globals()[b'plan'][b'clock_rounding_margin_ns'] = 1000
    return lua, proto, service


def test_actual_clock_value_reproduces_old_raw_double_echo_failure():
    lua, proto, service = rounded_service(old_service=True)
    obj = create(lua, proto, service)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    assert RAW != WIRE and obj[b'published_clocks'][1] == RAW
    command(lua, 2, 'run', 1, WIRE)
    obj[b'tick'](obj, RAW)
    assert obj[b'phase'] == b'failed'
    assert b'Host ACK must retain original source clock' in obj[b'failure']


def test_exact_actual_emitted_wire_clock_passes_without_tolerance_or_raw_timer_change():
    lua, proto, service = rounded_service()
    obj = create(lua, proto, service)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    raw = lua.globals()[b'rawSlot'](1)
    emitted = decode_slot(raw, raw, owner_id='a'*32, generation='engine:one')
    assert emitted.source_clock == WIRE
    assert json.loads(emitted.snapshot_json)['script_clock'] == WIRE
    assert obj[b'published_clocks'][1] == WIRE
    assert obj[b'published_raw_clocks'][1] == RAW
    assert obj[b'publisher'][b'last_source_clock'] == RAW
    command(lua, 2, 'run', 1, WIRE)
    obj[b'tick'](obj, RAW)
    assert obj[b'phase'] == b'publishing' and obj[b'committed_raw_source_clock'] == RAW
    assert obj[b'committed_source_clock'] == WIRE
    command(lua, 3, 'run', 1, math.nextafter(WIRE, math.inf))
    obj[b'tick'](obj, RAW)
    assert obj[b'phase'] == b'failed'  # even1ULP wrong echo is not tolerated


def test_rounded_clock_final_and_writer_fence_reach_sync_return():
    lua, proto, service = rounded_service()
    obj = create(lua, proto, service)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    command(lua, 2, 'retire', 1, WIRE)
    next_raw = math.nextafter(38.8736105, math.inf)
    lua.globals()[b'now'] = next_raw
    lua.globals()[b'snapshot'][b'script_clock'] = next_raw
    obj[b'tick'](obj, next_raw)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    assert obj[b'phase'] == b'final' and obj[b'final_source_clock'] == 38.8736105
    command(lua, 3, 'writer_fenced', 2, 38.8736105)
    lua.globals()[b'now'] = 38.9
    lua.globals()[b'snapshot'][b'script_clock'] = 38.9
    obj[b'tick'](obj, 38.9)
    assert obj[b'phase'] == b'sync_return' and lua.globals()[b'syncCount'] == 1
    assert lua.globals()[b'cleanupCount'] == 0


def test_original_raw_source_owns_two_second_timeout_even_when_wire_rounds_down():
    lua, proto, service = rounded_service()
    obj = create(lua, proto, service)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    command(lua, 2, 'run', 1, WIRE)
    obj[b'tick'](obj, RAW+2.)
    assert RAW+2.-WIRE > 2. and obj[b'phase'] == b'publishing'
    obj[b'tick'](obj, RAW+2.+1e-10)
    assert obj[b'phase'] == b'failed' and b'freshness timeout' in obj[b'failure']


def test_serialization_error_beyond_existing_margin_fails_before_slot_publication():
    lua, proto, service = rounded_service()
    lua.globals()[b'plan'][b'clock_rounding_margin_ns'] = 1

    def coarse(value):
        data = python_value(value)
        if 'source_clock' in data:
            data['source_clock'] = round(data['source_clock'], 2)
        if 'script_clock' in data:
            data['script_clock'] = round(data['script_clock'], 2)
        return json.dumps(data, separators=(',', ':')).encode()

    lua.globals()[b'json_python'] = coarse
    obj = create(lua, proto, service)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    assert obj[b'phase'] == b'failed'
    assert b'existing source-clock rounding margin' in obj[b'failure']
    assert lua.globals()[b'rawSlot'](1) == bytes(8212)


def test_terminal_failed_ack_after_join_is_preserved_and_reported_immediately():
    ack = (4, dict(phase='failed', reason='Host ACK must retain original source clock',
                   original_clock=38.648782, committed_source_clock=38.3720323))
    failures = []
    context = SimpleNamespace(control=SimpleNamespace(receive=lambda: ack), _fail=failures.append)
    with pytest.raises(RuntimeError, match='Host ACK must retain original source clock'):
        observe_sync_return(context)
    assert len(failures) == 1 and context.terminal_control_ack == dict(sequence=4, value=ack[1])


def test_new_builder_preserves_frozen_c159_and_emits_separate_versioned_sources():
    import hashlib

    from scripts.prepare_session_heartbeat_wire_v2 import PINNED, build

    outputs = build()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in PINNED.items())
    assert ROOT/'FlyBrainCamera-heartbeat-wire-v2.lua' in outputs
    assert all(('wire-v2' in str(path) or '_v2' in str(path)) for path in outputs)
    lua, _, _ = rounded_service()
    assert lua.eval(b'loadstring')(outputs[ROOT/'FlyBrainCamera-heartbeat-wire-v2.lua']) is not None
