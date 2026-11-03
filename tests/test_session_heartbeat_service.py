"""LuaJIT/FFI lifecycle fixture. No real mappings or live engine APIs."""

import json
from pathlib import Path

from test_session_heartbeat_lua_candidate import (
    no_fatal_signal_for_handled_luajit_errors,  # noqa: F401
    service,
)

from flybrain_drive.session_heartbeat_runtime import (
    CONTROL_SCHEMA,
    decode_control,
    encode_control,
)

SOURCE = Path('integrations/ac_csp/candidates/SessionHeartbeatService.lua')
OWNER, GEN, LEASE = 'a'*32, 'engine:one', 'b'*64


def fixture():
    lua, proto = service()

    def bytes_json(value):
        if isinstance(value, str):
            return value.encode()
        if isinstance(value, dict):
            return {bytes_json(k): bytes_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [bytes_json(v) for v in value]
        return value

    lua.globals()[b'json_parse_python'] = lambda text: lua.table_from(
        bytes_json(json.loads(text)), recursive=True)
    module = lua.execute(SOURCE.read_bytes())
    lua.globals()[b'serviceModule'] = module
    lua.execute(b'''
      JSON.parse=function(text) return json_parse_python(text) end
      hostControl=ffi.new('struct { '..serviceModule.CONTROL_LAYOUT..' }[1]')
      luaControl=ffi.new('struct { '..serviceModule.CONTROL_LAYOUT..' }[1]')
      local originalOpen=ac.writeMemoryMappedFile
      ac.readMemoryMappedFile=function(name,layout,persist)
        if pointerMode then return ffi.cast('struct { '..layout..' }*',hostControl) end
        return hostControl[0]
      end
      ac.writeMemoryMappedFile=function(name,layout,persist)
        if name:find('Control_',1,true) then
          if pointerMode then return ffi.cast('struct { '..layout..' }*',luaControl) end
          return luaControl[0]
        end
        return originalOpen(name,layout,persist)
      end
      quiescent=true; cleanupCount=0; syncCount=0
      callbacks={quiescent=function() return quiescent end,
        state=function() return snapshot end,
        cleanup=function(reason) cleanupCount=cleanupCount+1 end,
        sync_commit=function(state) syncCount=syncCount+1; return JSON.stringify(state) end}
      snapshot.epoch=3
      plan={schema='flybrain.session_heartbeat.arm.v1',owner_id=string.rep('a',32),
        generation='engine:one',lease_sha256=string.rep('b',64),epoch=3,
        clock_domain='host_perf_counter_QueryPerformanceCounter_ns',
        deadline_source_clock=59,deadline_qpc_ns=160000000000,
        registered_qpc_ns=100000000000,host_created_slots=true,max_snapshots=256}
      plan.host_ready=true
      lease={schema='flybrain.session_heartbeat.file_lease.v1',owner_id=plan.owner_id,
        generation=plan.generation,deadline_qpc_ns=plan.deadline_qpc_ns,
        registered_qpc_ns=plan.registered_qpc_ns,clock_domain=plan.clock_domain}
      setHost=function(raw) ffi.copy(hostControl,raw,#raw) end
      rawControl=function() return ffi.string(luaControl,ffi.sizeof(luaControl)) end
    ''')
    return lua, proto, module


def command(lua, number, action='run', committed=0, source=None):
    value = dict(schema=CONTROL_SCHEMA, owner_id=OWNER, generation=GEN,
                 lease_sha256=LEASE, command=action, committed_sequence=committed)
    if source is not None:
        value['committed_source_clock'] = source
    lua.globals()[b'setHost'](encode_control(value, number))


def create(lua, proto, module):
    command(lua, 1)
    obj = module[b'new'](proto, lua.globals()[b'plan'], lua.globals()[b'lease'],
                         lua.globals()[b'callbacks'])
    lua.globals()[b'transport'] = obj
    return obj


def test_camera_independent_commit_ack_and_quiescent_final_sync_handoff():
    lua, proto, module = fixture()
    obj = create(lua, proto, module)
    snap = lua.globals()[b'snapshot']
    assert obj[b'heartbeat'](obj, snap) is True
    command(lua, 2, 'retire', 1, 1.)
    lua.execute(b'now=1.25; snapshot.script_clock=1.25')
    obj[b'tick'](obj, 1.25)
    obj[b'heartbeat'](obj, snap)
    raw = lua.globals()[b'rawControl']()
    _, final = decode_control(raw, raw, owner_id=OWNER, generation=GEN, lease_sha256=LEASE)
    assert final['phase'] == 'final' and final['final_sequence'] == 2
    command(lua, 3, 'writer_fenced', 2, 1.25)
    lua.execute(b'now=1.5; snapshot.script_clock=1.5')
    obj[b'tick'](obj, 1.5)
    assert obj[b'phase'] == b'sync_return' and lua.globals()[b'syncCount'] == 1
    assert obj[b'heartbeat'](obj, snap) is True
    held_ack = lua.globals()[b'rawControl']()
    lua.execute(b'now=1.8; snapshot.script_clock=1.8')
    obj[b'tick'](obj, 1.8)
    assert obj[b'heartbeat'](obj, snap, True) is True
    assert lua.globals()[b'syncCount'] == 1 and lua.globals()[b'rawControl']() == held_ack
    command(lua, 4, 'release_confirmed', 2, 1.25)
    obj[b'tick'](obj, 1.8)
    assert obj[b'closed'] is True and lua.globals()[b'cleanupCount'] == 0


def test_host_ack_delivery_does_not_rebase_original_source_freshness():
    lua, proto, module = fixture()
    obj = create(lua, proto, module)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    command(lua, 2, 'run', 1, 1.)
    lua.globals()[b'now'] = 3.01
    obj[b'tick'](obj, 3.01)
    assert obj[b'phase'] == b'failed' and lua.globals()[b'cleanupCount'] == 1
    obj[b'tick'](obj, 3.02)
    assert lua.globals()[b'cleanupCount'] == 1
    assert obj[b'heartbeat'](obj, lua.globals()[b'snapshot']) is True
    assert lua.globals()[b'syncCount'] == 0


def test_writer_fence_before_final_and_foreign_commit_trigger_original_cleanup():
    lua, proto, module = fixture()
    obj = create(lua, proto, module)
    command(lua, 2, 'writer_fenced')
    obj[b'tick'](obj, 1.)
    assert obj[b'phase'] == b'failed' and lua.globals()[b'syncCount'] == 0
    assert lua.globals()[b'cleanupCount'] == 1


def test_retirement_waits_for_original_quiescence_instead_of_dropping_control_state():
    lua, proto, module = fixture()
    obj = create(lua, proto, module)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    command(lua, 2, 'retire', 1, 1.)
    lua.execute(b'now=1.25; snapshot.script_clock=1.25; quiescent=false')
    obj[b'tick'](obj, 1.25)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    assert obj[b'phase'] == b'publishing' and lua.globals()[b'syncCount'] == 0


def test_forced_original_task_refresh_is_not_suppressed_inside_quarter_second():
    lua, proto, module = fixture()
    obj = create(lua, proto, module)
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'])
    lua.execute(b'now=1.05; snapshot.script_clock=1.05; '
                b"snapshot.external_task.phase='stop0'; snapshot.external_task.desired_speed_m_s=0")
    obj[b'heartbeat'](obj, lua.globals()[b'snapshot'], True)
    assert obj[b'publisher'][b'published_count'] == 2
    assert obj[b'published_clocks'][2] == 1.05
    # Repeated reset remains bounded by the original finite snapshot cap.
    assert obj[b'publisher'][b'max_snapshots'] == 256


def test_session_epoch_change_fails_without_rebase_or_synchronous_fallback():
    lua, proto, module = fixture()
    obj = create(lua, proto, module)
    lua.execute(b'snapshot.epoch=4')
    assert obj[b'heartbeat'](obj, lua.globals()[b'snapshot']) is True
    assert obj[b'phase'] == b'failed' and lua.globals()[b'cleanupCount'] == 1
    assert obj[b'publisher'][b'published_count'] == 0 and lua.globals()[b'syncCount'] == 0


def test_installed_sdk_control_reader_supports_struct_pointer_as_well_as_struct_value():
    lua, proto, module = fixture()
    lua.globals()[b'pointerMode'] = True
    obj = create(lua, proto, module)
    assert obj[b'heartbeat'](obj, lua.globals()[b'snapshot']) is True
    command(lua, 2, 'run', 1, 1.)
    obj[b'tick'](obj, 1.1)
    assert obj[b'phase'] == b'publishing' and obj[b'committed_sequence'] == 1
    raw = lua.globals()[b'rawControl']()
    assert decode_control(raw, raw, owner_id=OWNER, generation=GEN, lease_sha256=LEASE)


def test_full_candidate_is_compilable_and_builder_preserves_pinned_base():
    import hashlib

    from scripts.prepare_session_heartbeat_lua import BASE, BASE_SHA256, build

    lua, _, _ = fixture()
    content = build()
    # Compile only: no top-level service/camera operations execute.
    assert lua.eval(b'loadstring')(content) is not None
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA256
    assert b'local heartbeatOrphanGuard = io.fileExists(heartbeatLeasePath)' in content
    original = BASE.read_text(encoding='utf8')
    candidate = content.decode().replace('\r\n', '\n')
    start = 'local function stopExpertProbe('
    end = 'local function pollHeartbeatTransport('
    block_end = original.index('local function pollSession(now)')
    original_block = original[original.index(start):block_end]
    assert original_block in candidate[:candidate.index(end)]


def test_full_poll_session_holds_sync_bytes_and_blocks_requests_until_host_release(monkeypatch):
    from test_ac_lua_session_receipts import service as full_service

    from scripts.prepare_session_heartbeat_lua import BASE, build

    content = build().decode()+'\nfunction testInjectTransport(t) heartbeatTransport=t end\n'
    original_read = Path.read_text

    def candidate(path, *args, **kwargs):
        return content if path == BASE else original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', candidate)
    lua, _ = full_service()
    lua.execute('''
      frozenStatus={script_clock=1.5,exact_ack_bound_status=true}
      files['mock/flybrain/csp-session/status.json']=frozenStatus
      hbCalls=0
      testInjectTransport({phase='sync_return',committed_sequence=2,
        tick=function() end, heartbeat=function() hbCalls=hbCalls+1;return true end})
      now=1.6;script.update(0)
      now=1.9;script.update(0)
      assert(files['mock/flybrain/csp-session/status.json']==frozenStatus)
      assert(calls==0 and hbCalls==2)
    ''')


def test_full_glue_accepts_fresh_owner_only_after_closed_and_never_replays_old_plan(monkeypatch):
    from test_ac_lua_session_receipts import service as full_service

    from scripts.prepare_session_heartbeat_lua import BASE, build

    content = build().decode()+'''
      function testClosedTransport()
        heartbeatTransport={phase='closed',tick=function() end}
        heartbeatArmOwner=string.rep('a',32)
        heartbeatService.new=function(proto,plan,lease,callbacks)
          newOwnerCount=(newOwnerCount or 0)+1;newOwner=plan.owner_id
          return {phase='armed',committed_sequence=0,tick=function() end,
            heartbeat=function() end}
        end
        heartbeatProtocol.crc32=function() return 1234 end
      end
    '''
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k:
                        content if path == BASE else original_read(path, *a, **k))
    lua, _ = full_service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      testClosedTransport();script.update(0)
      files['mock/flybrain/csp-session/status.json.flybrain-heartbeat-owner']={
        schema='flybrain.session_heartbeat.file_lease.v1',owner_id=string.rep('b',32),
        generation='17:1',clock_domain='host_perf_counter_QueryPerformanceCounter_ns',
        registered_qpc_ns=100000000000,deadline_qpc_ns=160000000000}
      files['mock/flybrain/csp-session/heartbeat-transport.json']={
        schema='flybrain.session_heartbeat.arm.v1',generation='17:1',epoch=0,
        owner_id=string.rep('a',32),host_ready=true,lease_crc32=1234,lease_sha256=string.rep('c',64),
        registered_qpc_ns=100000000000,deadline_qpc_ns=160000000000}
      now=1.2;script.update(0);assert(newOwnerCount==nil)
      files['mock/flybrain/csp-session/heartbeat-transport.json'].owner_id=string.rep('b',32)
      now=1.4;script.update(0)
      assert(newOwnerCount==1 and newOwner==string.rep('b',32) and calls==0)
    ''')


def test_probe_callback_boundaries_preserve_single_call_and_old_postcall_clock(monkeypatch):
    from test_ac_lua_session_receipts import service as full_service

    from scripts.prepare_session_heartbeat_lua import BASE, build

    content = build().decode()+'''
      function testProbeTiming(callback)
        expertProbe={calls={}}
        probeCall('fixture',callback)
        return expertProbe.calls[1]
      end
    '''
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k:
                        content if path == BASE else original_read(path, *a, **k))
    lua, _ = full_service()
    lua.execute('''
      callbackCount=0
      record=testProbeTiming(function()
        callbackCount=callbackCount+1;callBegan=now;now=now+.05;callEnded=now
      end)
      assert(record.started_clock<=callBegan and callEnded<=record.finished_clock)
      assert(record.finished_clock<=record.script_clock and record.returned==true)
      assert(callbackCount==1 and record.error==nil)
      failed=testProbeTiming(function()
        callbackCount=callbackCount+1;now=now+.03;error('uncertain original callback')
      end)
      assert(failed.returned==false and failed.error:find('uncertain original callback',1,true))
      assert(failed.started_clock<failed.finished_clock and callbackCount==2)
    ''')


def full_startup_fixture(monkeypatch, *, orphan=False):
    from test_ac_lua_session_receipts import service as full_service

    from scripts.prepare_session_heartbeat_lua import BASE, build

    prelude = "files['mock/flybrain/csp-session/status.json.flybrain-heartbeat-owner']={}\n"
    content = (prelude if orphan else '')+build().decode()+'''
      function testHeartbeatError() return sessionError end
      function testHeartbeatState()
        return heartbeatTransport and heartbeatTransport.phase or 'waiting',
          heartbeatPending and heartbeatPending.deadline_clock
      end
      function testStartupFactory()
        heartbeatProtocol.crc32=function() return 1234 end
        heartbeatService.new=function(proto,plan,lease,callbacks)
          startupCount=(startupCount or 0)+1
          return {phase='armed',committed_sequence=0,tick=function() end,
            heartbeat=function() end}
        end
      end
      function testKnownClosedLease(lease)
        heartbeatTransport={phase='closed',tick=function() end}
        heartbeatActiveLeaseText=lease
        heartbeatArmOwner=lease.owner_id
      end
    '''
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k:
                        content if path == BASE else original_read(path, *a, **k))
    lua, _ = full_service()
    lua.execute('''
      testStartupFactory()
      leasePath='mock/flybrain/csp-session/status.json.flybrain-heartbeat-owner'
      armPath='mock/flybrain/csp-session/heartbeat-transport.json'
      validLease={schema='flybrain.session_heartbeat.file_lease.v1',owner_id=string.rep('b',32),
        generation='17:1',clock_domain='host_perf_counter_QueryPerformanceCounter_ns',
        registered_qpc_ns=100000000000,deadline_qpc_ns=160000000000}
      validArm={schema='flybrain.session_heartbeat.arm.v1',generation='17:1',epoch=0,
        owner_id=string.rep('b',32),host_ready=true,lease_crc32=1234,lease_sha256=string.rep('c',64),
        registered_qpc_ns=100000000000,deadline_qpc_ns=160000000000}
      frozenStatus={exact_old_sync=true};files['mock/flybrain/csp-session/status.json']=frozenStatus
    ''')
    return lua


def test_lease_before_ready_arm_waits_through_partial_absent_stale_without_poison(monkeypatch):
    lua = full_startup_fixture(monkeypatch)
    lua.execute('''
      local originalParse=JSON.parse
      JSON.parse=function(value)
        if value=='partial JSON' then error('fixture incomplete JSON') end
        return originalParse(value)
      end
      files[leasePath]='partial JSON';now=1.1;script.update(0)
      assert(testHeartbeatError()==nil and startupCount==nil and calls==0)
      files[leasePath]=validLease;now=1.3;script.update(0)
      assert(testHeartbeatError()==nil and startupCount==nil and calls==0)
      files[armPath]={owner_id=string.rep('a',32)};now=1.5;script.update(0)
      assert(testHeartbeatError()==nil and startupCount==nil and calls==0)
      files[armPath]=validArm;validArm.host_ready=false;now=1.7;script.update(0)
      assert(testHeartbeatError()==nil and startupCount==nil and calls==0)
      assert(files['mock/flybrain/csp-session/status.json']==frozenStatus)
      validArm.host_ready=true;now=1.9;script.update(0)
      assert(testHeartbeatError()==nil and startupCount==1 and calls==0)
    ''')


def test_matching_ready_plan_with_wrong_raw_lease_crc_fails_closed(monkeypatch):
    lua = full_startup_fixture(monkeypatch)
    lua.execute('''
      files[leasePath]=validLease;files[armPath]=validArm;validArm.lease_crc32=99
      now=1.1;script.update(0)
      assert(testHeartbeatError():find('binding differs',1,true) and startupCount==nil and calls==0)
    ''')


def test_ready_plan_wait_deadline_never_rebases_and_late_plan_cannot_retry(monkeypatch):
    lua = full_startup_fixture(monkeypatch)
    lua.execute('''
      files[leasePath]=validLease;now=1.1;script.update(0)
      now=2.5;files[armPath]={owner_id=string.rep('a',32)};script.update(0)
      local phase,deadline=testHeartbeatState();assert(deadline==3.1)
      now=3.11;script.update(0)
      assert(testHeartbeatError():find('timeout',1,true) and startupCount==nil and calls==0)
      files[armPath]=validArm;now=3.3;script.update(0)
      assert(startupCount==nil and calls==0)
    ''')


def test_existing_lease_at_lua_load_stays_orphan_fenced_even_if_later_removed(monkeypatch):
    lua = full_startup_fixture(monkeypatch, orphan=True)
    lua.execute('''
      files[leasePath]=validLease;files[armPath]=validArm;script.update(0)
      assert(testHeartbeatError():find('orphan',1,true) and startupCount==nil and calls==0)
      files[leasePath]=nil;now=1.5;script.update(0)
      assert(startupCount==nil and calls==0)
    ''')


def test_release_confirmed_before_unlink_keeps_known_completed_owner_out_of_new_startup(
        monkeypatch):
    lua = full_startup_fixture(monkeypatch)
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      files[leasePath]=validLease;testKnownClosedLease(validLease)
      now=1.1;script.update(0);now=1.4;script.update(0)
      assert(testHeartbeatError()==nil and startupCount==nil and calls==0)
      local phase,deadline=testHeartbeatState();assert(deadline==nil)
      assert(files['mock/flybrain/csp-session/status.json']==frozenStatus)
      files[leasePath]=nil;now=1.6;script.update(0);now=3.4;script.update(0)
      assert(testHeartbeatError()==nil and startupCount==nil and calls==0)
    ''')


def test_stable_foreign_lease_fails_closed_during_wait(monkeypatch):
    lua = full_startup_fixture(monkeypatch)
    lua.execute('''
      validLease.generation='foreign';files[leasePath]=validLease
      now=1.1;script.update(0)
      assert(testHeartbeatError():find('foreign',1,true) and startupCount==nil and calls==0)
    ''')


def test_sync_return_blocks_new_camera_begin_until_closed_without_blocking_update(monkeypatch):
    from test_ac_lua_session_receipts import service as full_service

    from scripts.prepare_session_heartbeat_lua import BASE, build

    content = build().decode()+'''
      function testHoldCamera()
        heartbeatTransport={phase='sync_return',committed_sequence=2,
          tick=function() end,heartbeat=function() end}
        begin=function() beginCount=(beginCount or 0)+1 end
      end
      function testCloseTransport() heartbeatTransport.phase='closed' end
    '''
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k:
                        content if path == BASE else original_read(path, *a, **k))
    lua, _ = full_service()
    lua.execute('''
      files['mock/flybrain/csp-camera/request.json']={id='queued-camera'}
      files['mock/flybrain/csp-session/request.json']=nil
      testHoldCamera();now=1.1;script.update(0);now=1.4;script.update(0)
      assert(beginCount==nil and calls==0)
      testCloseTransport();now=1.7;script.update(0)
      assert(beginCount==1 and calls==0)
    ''')
