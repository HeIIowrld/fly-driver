"""CPU LuaJIT/FFI protocol tests; synthetic arrays only, never Windows mappings."""

import faulthandler
import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

from flybrain_drive.session_heartbeat_candidate import decode_slot

lua_module = pytest.importorskip('lupa.luajit21')
SOURCE = Path('integrations/ac_csp/candidates/SessionHeartbeatMMF.lua')
HEADER = struct.Struct('<5I')
CAPACITY = 8192


@pytest.fixture(autouse=True)
def no_fatal_signal_for_handled_luajit_errors():
    restore = sys.platform == 'win32' and faulthandler.is_enabled()
    if restore:
        faulthandler.disable()
    try:
        yield
    finally:
        if restore:
            faulthandler.enable()


def python_value(value):
    if isinstance(value, bytes):
        return value.decode('utf8')
    if hasattr(value, 'items'):
        return {python_value(k): python_value(v) for k, v in value.items()}
    return value


def service():
    lua = lua_module.LuaRuntime(unpack_returned_tuples=True, encoding=None)
    lua.globals()[b'json_python'] = lambda value: json.dumps(
        python_value(value), separators=(',', ':'), ensure_ascii=False).encode('utf8')
    lua.execute(b'''
      ffi=require('ffi'); now=1; opened={}; disposed={}; slots={}
      os.preciseClock=function() return now end
      JSON={stringify=function(value) return json_python(value) end}
      ac={writeMemoryMappedFile=function(name,layout,persist)
        table.insert(opened,{name=name,persist=persist})
        assert(slots[name], 'Synthetic host must precreate slots')
        return slots[name][0]
      end,disposeMemoryMappedFile=function(reference) table.insert(disposed,reference) end}
      -- Every non-protocol engine or filesystem operation is deliberately absent.
      camera=nil; io.save=nil; io.saveAsync=nil
    ''')
    proto = lua.execute(SOURCE.read_bytes())
    lua.globals()[b'proto'] = proto
    lua.execute(b'''
      config={owner_id=string.rep('a',32),generation='engine:one',duration_seconds=60,
        max_snapshots=256,host_created_slots=true}
      for index=0,1 do
        slots[proto.channel_name(config.owner_id,index)]=ffi.new('struct { '..proto.LAYOUT..' }[1]')
      end
      snapshot={schema='flybrain.csp.session.status.v1',generation='engine:one',
        script_clock=1,paused=false,ai_controlled=true,camera_active=false,
        external_task={desired_speed_m_s=27.777777777778,phase='accelerate100',
          source='original task',actor_input=true},unknown_sensor_valid=false}
      rawSlot=function(index)
        local array=slots[proto.channel_name(config.owner_id,index)]
        return ffi.string(array,ffi.sizeof(array))
      end
    ''')
    return lua, proto


def new(lua, proto):
    publisher = proto[b'new'](lua.globals()[b'config'])
    lua.globals()[b'publisher'] = publisher
    return publisher


def publish(publisher, snapshot):
    return publisher[b'publish'](publisher, snapshot)


def decoded(raw):
    magic, version, sequence, length, crc = HEADER.unpack_from(raw)
    assert len(raw) == HEADER.size+CAPACITY
    assert (magic, version) == (0x46424832, 1)
    assert sequence % 2 == 0 and sequence > 0
    payload = raw[HEADER.size:HEADER.size+length]
    assert zlib.crc32(payload) == crc
    return json.loads(payload), sequence//2


@pytest.mark.parametrize('payload', [b'', b'123456789', bytes(range(256)),
                                     '한글 original'.encode()])
def test_crc32_matches_python_including_unsigned_high_bit(payload):
    lua, proto = service()
    assert len(lua.globals()[b'opened']) == 0
    assert proto[b'crc32'](payload) == zlib.crc32(payload)


def test_exact_two_slot_layout_preserves_original_snapshot_and_source_clock():
    lua, proto = service()
    publisher = new(lua, proto)
    assert len(lua.globals()[b'opened']) == 2
    assert all(lua.globals()[b'opened'][i][b'persist'] is False for i in (1, 2))
    snapshot = lua.globals()[b'snapshot']
    original = python_value(snapshot)
    assert publish(publisher, snapshot) == 1
    envelope, sequence = decoded(lua.globals()[b'rawSlot'](1))
    assert sequence == envelope['sequence'] == 1
    assert envelope['schema'] == 'flybrain.session_heartbeat.packet.v1'
    assert envelope['owner_id'] == 'a'*32 and envelope['generation'] == 'engine:one'
    assert envelope['source_clock'] == original['script_clock'] == 1
    assert json.loads(envelope['snapshot_json']) == original
    snapshot[b'paused'] = True
    assert json.loads(envelope['snapshot_json'])['paused'] is False
    lua.globals()[b'now'] = 1.25
    snapshot[b'script_clock'] = 1.2
    assert publish(publisher, snapshot) == 2
    newer, sequence = decoded(lua.globals()[b'rawSlot'](0))
    assert sequence == 2 and newer['source_clock'] == 1.2
    assert newer['snapshot_json'] != envelope['snapshot_json']
    assert publisher[b'record'](publisher)[b'camera_independent'] is True


def test_period_budget_and_expiry_refuse_without_rewriting_an_old_packet():
    lua, proto = service()
    lua.globals()[b'config'][b'max_snapshots'] = 2
    publisher = new(lua, proto)
    snapshot = lua.globals()[b'snapshot']
    assert publish(publisher, snapshot) == 1
    previous = lua.globals()[b'rawSlot'](1)
    lua.globals()[b'now'] = 1.1
    assert publish(publisher, snapshot) == (None, b'not_due')
    assert lua.globals()[b'rawSlot'](1) == previous
    lua.globals()[b'now'] = 1.25
    snapshot[b'script_clock'] = 1.25
    assert publish(publisher, snapshot) == 2
    lua.globals()[b'now'] = 1.5
    assert publish(publisher, snapshot) == (None, b'budget_exhausted')
    assert publisher[b'retired'] is True
    with pytest.raises(lua_module.LuaError, match='retired'):
        publish(publisher, snapshot)
    publisher[b'close'](publisher)
    publisher[b'close'](publisher)
    assert len(lua.globals()[b'disposed']) == 2
    lua, proto = service()
    lua.globals()[b'config'][b'duration_seconds'] = .25
    publisher = new(lua, proto)
    lua.globals()[b'now'] = 1.251
    assert publish(publisher, lua.globals()[b'snapshot']) == (None, b'expired')
    assert publisher[b'published_count'] == 0


def test_source_clock_origin_change_is_latched_and_second_open_failure_closes_only_its_reference():
    lua, proto = service()
    publisher = new(lua, proto)
    lua.globals()[b'now'] = .5
    with pytest.raises(lua_module.LuaError, match='changed origin'):
        publish(publisher, lua.globals()[b'snapshot'])
    assert publisher[b'retired'] and publisher[b'last_error']
    lua, proto = service()
    lua.execute(b'''
      local open=ac.writeMemoryMappedFile
      ac.writeMemoryMappedFile=function(name,layout,persist)
        if name:sub(-2)=='_1' then error('injected second open failure') end
        return open(name,layout,persist)
      end
    ''')
    with pytest.raises(lua_module.LuaError, match='second open failure'):
        new(lua, proto)
    assert len(lua.globals()[b'disposed']) == 1


@pytest.mark.parametrize('mutation', ["config.owner_id=string.rep('A',32)",
    'config.host_created_slots=false', "config.generation=''", 'config.duration_seconds=61',
    'config.max_snapshots=257', 'config.max_snapshots=2.5'])
def test_invalid_config_never_opens_a_slot(mutation):
    lua, proto = service()
    lua.execute(mutation.encode())
    with pytest.raises(lua_module.LuaError):
        new(lua, proto)
    assert len(lua.globals()[b'opened']) == 0


@pytest.mark.parametrize('field,value', [(b'magic', 1), (b'sequence', 2), (b'length', 1),
                                       (b'crc32', 19)])
def test_existing_map_bytes_are_never_reset_even_when_other_slot_opened_first(field, value):
    lua, proto = service()
    lua.execute(b'''
      name=proto.channel_name(config.owner_id,1)
      slots[name]=ffi.new('struct { '..proto.LAYOUT..' }[1]')
    ''')
    lua.globals()[b'changedField'], lua.globals()[b'changedValue'] = field, value
    lua.execute(b'slots[name][0][changedField]=changedValue')
    before = lua.globals()[b'rawSlot'](1)
    with pytest.raises(lua_module.LuaError, match='not pristine'):
        new(lua, proto)
    assert lua.globals()[b'rawSlot'](1) == before
    assert len(lua.globals()[b'disposed']) == 2


@pytest.mark.parametrize('mutation', ["snapshot.generation='wrong'", 'snapshot.script_clock=0/0',
    'snapshot.script_clock=math.huge', 'snapshot.script_clock=-1', 'snapshot.script_clock=1.001',
    "snapshot.schema='wrong'"])
def test_invalid_original_clock_or_generation_retires_without_repair_or_publication(mutation):
    lua, proto = service()
    publisher = new(lua, proto)
    lua.execute(mutation.encode())
    with pytest.raises(lua_module.LuaError):
        publish(publisher, lua.globals()[b'snapshot'])
    assert publisher[b'status'] == b'failed' and publisher[b'retired'] is True
    assert publisher[b'published_count'] == 0
    assert lua.globals()[b'rawSlot'](0) == bytes(HEADER.size+CAPACITY)
    assert lua.globals()[b'rawSlot'](1) == bytes(HEADER.size+CAPACITY)


def test_source_clock_regression_and_oversized_json_never_overwrite_last_valid_snapshot():
    lua, proto = service()
    publisher = new(lua, proto)
    snapshot = lua.globals()[b'snapshot']
    assert publish(publisher, snapshot) == 1
    before = lua.globals()[b'rawSlot'](1)
    lua.execute(b'now=1.25; snapshot.script_clock=.9')
    with pytest.raises(lua_module.LuaError, match='strictly increasing'):
        publish(publisher, snapshot)
    assert lua.globals()[b'rawSlot'](1) == before
    lua, proto = service()
    publisher = new(lua, proto)
    lua.execute(b"snapshot.oversized=string.rep('x',6144)")
    with pytest.raises(lua_module.LuaError, match='snapshot JSON'):
        publish(publisher, lua.globals()[b'snapshot'])
    assert publisher[b'published_count'] == 0


def test_repeated_original_source_clock_cannot_be_republished_with_a_new_sequence():
    lua, proto = service()
    publisher = new(lua, proto)
    snapshot = lua.globals()[b'snapshot']
    assert publish(publisher, snapshot) == 1
    original = lua.globals()[b'rawSlot'](1)
    lua.globals()[b'now'] = 1.25
    with pytest.raises(lua_module.LuaError, match='strictly increasing'):
        publish(publisher, snapshot)
    assert publisher[b'published_count'] == 1 and publisher[b'retired'] is True
    assert lua.globals()[b'rawSlot'](1) == original
    assert lua.globals()[b'rawSlot'](0) == bytes(HEADER.size+CAPACITY)


def test_actual_python_decoder_accepts_original_lua_ffi_bytes_and_snapshot_json():
    lua, proto = service()
    publisher = new(lua, proto)
    snapshot = lua.globals()[b'snapshot']
    original = python_value(snapshot)
    for sequence, clock, slot in [(1, 1., 1), (2, 1.25, 0)]:
        lua.globals()[b'now'] = clock
        snapshot[b'script_clock'] = clock
        assert publish(publisher, snapshot) == sequence
        raw = lua.globals()[b'rawSlot'](slot)
        received = decode_slot(raw, raw, owner_id='a'*32, generation='engine:one')
        assert received.sequence == sequence and received.source_clock == clock
        original['script_clock'] = clock
        assert json.loads(received.snapshot_json) == original


def test_envelope_expansion_and_serialization_failure_are_not_truncated_or_retried():
    lua, proto = service()
    publisher = new(lua, proto)
    lua.globals()[b'snapshot'][b'backslashes'] = b'\\'*2500
    with pytest.raises(lua_module.LuaError, match='envelope'):
        publish(publisher, lua.globals()[b'snapshot'])
    assert publisher[b'published_count'] == 0
    lua, proto = service()
    publisher = new(lua, proto)
    lua.execute(b"JSON.stringify=function() error('injected serialize failure') end")
    with pytest.raises(lua_module.LuaError, match='serialize failure'):
        publish(publisher, lua.globals()[b'snapshot'])
    assert publisher[b'last_error'] and publisher[b'retired']


def test_seqlock_is_odd_for_every_header_mutation_and_even_only_after_valid_payload_crc():
    lua, proto = service()
    lua.execute(b'''
      observedOdd,committed=0,0
      ac.writeMemoryMappedFile=function(name,layout,persist)
        local values={magic=0,version=0,sequence=0,length=0,crc32=0,
          payload=ffi.new('char[8192]')}
        return setmetatable({}, {__index=values,__newindex=function(_,key,value)
          if key=='sequence' then
            if value%2==1 then observedOdd=observedOdd+1
            else
              assert(values.sequence%2==1)
              assert(values.magic==proto.MAGIC and values.version==proto.VERSION)
              assert(proto.crc32(ffi.string(values.payload,values.length))==values.crc32)
              committed=committed+1
            end
          else assert(values.sequence%2==1) end
          values[key]=value
        end})
      end
    ''')
    publisher = new(lua, proto)
    assert publish(publisher, lua.globals()[b'snapshot']) == 1
    assert lua.globals()[b'observedOdd'] == lua.globals()[b'committed'] == 1
