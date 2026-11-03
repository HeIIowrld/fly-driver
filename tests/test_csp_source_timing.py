"""Execute opt-in producer timing with injected clock/I/O delays, never a game."""

import copy
import inspect
import json
from types import SimpleNamespace

import pytest
from test_ac_lua_session_receipts import (
    fast_motion_service,
    handled_luajit_errors_are_not_python_fatal_signals,  # noqa: F401
    service,
    set_owned_camera,
)

from flybrain_drive.ac_csp_vision import CSPVisionRecorder, validate_source_timing_ledger


def timed_camera(*, motion=False):
    lua, _ = fast_motion_service() if motion else service()
    lua.globals().keepMotion = motion
    lua.execute("request=files['mock/flybrain/csp-session/request.json']")
    set_owned_camera(lua)
    lua.execute('''
      if not keepMotion then files['mock/flybrain/csp-session/request.json']=nil end
      function upvalue(fn,key)
        for i=1,60 do
          local name,value=debug.getupvalue(fn,i)
          if name==key then return value,i end
        end
        error('upvalue missing: '..key)
      end
      ledger={schema='flybrain.csp.source_timing.v1', request_id='owned-camera',
        generation='17:1', directory='mock-camera', started_clock=now,
        clock_domain='csp_os_preciseClock_seconds', max_records=131072,
        record_fields={'stage','started_clock','finished_clock','duration_seconds',
          'succeeded','update_index','capture_index','detail'},
        overflow_records=0,update_index=0,records={},actor_input=false,
        timing_observation_only=true,per_frame_timing_disk_writes=0}
      local _,index=upvalue(script.update,'sourceTiming')
      debug.setupvalue(script.update,index,ledger)
      camera.source_timing=ledger
      originalSave=io.save
      io.save=function(path,value,ensure)
        if path=='mock/flybrain/csp-session/status.json' then now=now+.182 end
        return originalSave(path,value,ensure)
      end
    ''')
    if not motion:
        lua.execute('sim.isPaused=true; car.isAIControlled=false')
    return lua


def lua_rows(lua):
    rows = lua.globals().ledger.records
    fields = [lua.globals().ledger.record_fields[i] for i in range(1, 9)]
    return [{name: rows[i][j] for j, name in enumerate(fields, 1)}
            for i in range(1, len(rows)+1)]


def test_timing_records_blocked_call_and_update_but_flushes_only_after_capture_closes():
    lua = timed_camera()
    lua.execute('script.update(0)')
    assert lua.eval("files['mock-camera/source-timing-v1.json']") is None
    rows = lua_rows(lua)
    save = next(row for row in rows if row['stage'] == 'write.io.save'
                and row.get('detail') == 'mock/flybrain/csp-session/status.json')
    assert save['duration_seconds'] == pytest.approx(.182)
    heartbeat = next(row for row in rows if row['stage'] == 'session_heartbeat')
    assert heartbeat['started_clock'] <= save['started_clock']
    assert heartbeat['finished_clock'] >= save['finished_clock']
    assert {row['stage'] for row in rows} >= {
        'sessionState', 'write.JSON.stringify', 'pollExpertProbe', 'pollSession',
        'clock_ping.io.load', 'stop.io.fileExists', 'script.update'}
    lua.execute("files['mock-camera/stop']=true; now=1.4; script.update(0)")
    assert lua.eval("files['mock-camera/result.json'].status") == 'stopped'
    assert lua.eval("files['mock-camera/source-timing-v1.json'].flushed_after_capture")
    assert lua.eval('ledger.capture_closed_clock <= ledger.finished_clock')
    assert lua.eval('ledger.record_count == #ledger.records')
    assert lua.eval('calls') == 0


def test_default_has_no_timing_ledger_and_preserves_lifecycle_side_effects():
    lua, _ = service()
    lua.execute('script.update(0); now=1.3; script.update(0)')
    assert lua.eval('calls') == 1
    assert not any('source-timing' in path for path in lua.globals().files.keys())
    assert inspect.signature(CSPVisionRecorder).parameters['source_timing_v1'].default is False


def test_capture_preamble_and_first_render_separate_injected_delays_then_ack_closes():
    lua = timed_camera()
    lua.execute('''
      vectorMeta={}
      vectorMeta.__add=function(a,b) return setmetatable({x=0,y=0,z=0},vectorMeta) end
      vectorMeta.__sub=vectorMeta.__add; vectorMeta.__mul=vectorMeta.__add
      vectorMeta.__unm=function(a) return a end
      vec3=function() return setmetatable({x=0,y=0,z=0},vectorMeta) end
      car.position=vec3(); car.look=vec3(); car.side=vec3(); car.up=vec3()
      car.bodyTransform={transformPoint=function(self,v) return v end}
      ac.INIConfig={carData=function()
        now=now+.1
        return {sections={DASH_CAM={POS={0,0,0}}},get=function() return vec3() end}
      end}
      camera.next=0; camera.frames=0; camera.request.frames=1
      camera.request.image_format='bmp'; camera.request.shared_transport=true
      camera.request.period_seconds=.04; camera.request.exposure=.35
      for _,id in ipairs({'front','left','right','rear'}) do
        table.insert(camera.views,{id=id,shot={
          update=function() now=now+.002 end,sharedHandle=function() return '19ULL' end,
          viewMatrix=function() return 'view' end,
          projectionMatrix=function() return 'projection' end,
          dispose=function() end}})
      end
      script.update(0)
    ''')
    rows = lua_rows(lua)
    ini = next(row for row in rows if row['stage'] == 'capture.INIConfig.carData')
    render = next(row for row in rows if row['stage'] == 'capture.first_render')
    preamble = next(row for row in rows if row['stage'] == 'capture.preamble_to_first_render')
    assert ini['duration_seconds'] == pytest.approx(.1)
    assert preamble['duration_seconds'] >= .1-1e-10
    assert render['duration_seconds'] == pytest.approx(.002)
    assert lua.eval("files['mock-camera/source-timing-v1.json']") is None
    lua.execute('''
      files['mock-camera/ack-0000.json']={request_id='owned-camera',generation='17:1',capture_id=0}
      now=1.5; script.update(0)
    ''')
    assert lua.eval("files['mock-camera/result.json'].status") == 'completed'
    stages = {row['stage'] for row in lua_rows(lua)}
    assert {'ACK.io.load', 'ACK.JSON.parse'} <= stages
    assert lua.eval("files['mock-camera/source-timing-v1.json'].final_capture_count") == 1


def test_fixed_memory_ceiling_reports_overflow_without_reset_or_timing_io():
    lua = timed_camera()
    lua.execute('''
      for i=1,131072 do ledger.records[i]={'retained-original',1,1,0,true,0,false,false} end
      script.update(0)
    ''')
    assert lua.eval('#ledger.records') == 131072
    assert lua.eval('ledger.overflow_records') > 0
    assert lua.eval("files['mock-camera/source-timing-v1.json']") is None
    assert lua.eval('calls') == 0


def test_heartbeat_write_failure_is_observed_and_legacy_recovery_stays_intact():
    lua = timed_camera()
    lua.execute("failures['mock/flybrain/csp-session/status.json']=1; script.update(0)")
    failed = [row for row in lua_rows(lua) if row['stage'] == 'write.io.save'
              and row.get('detail') == 'mock/flybrain/csp-session/status.json']
    assert failed[0]['succeeded'] is False
    lua.execute('now=1.4; script.update(0)')
    assert lua.eval("files['mock/flybrain/csp-session/status.json'].heartbeat_write_failures") == 1
    assert lua.eval('calls') == 0


def test_terminal_timing_write_failure_never_retries_capture_or_controls():
    lua = timed_camera()
    lua.execute('''
      flushAttempts=0
      local save=io.save
      io.save=function(path,value,ensure)
        if path=='mock-camera/source-timing-v1.json' then
          flushAttempts=flushAttempts+1; return false
        end
        return save(path,value,ensure)
      end
      files['mock-camera/stop']=true
      script.update(0); now=2; script.update(0)
    ''')
    assert lua.eval('flushAttempts') == 1
    assert lua.eval("files['mock-camera/source-timing-v1.json']") is None
    assert lua.eval("files['mock-camera/result.json'].status") == 'stopped'
    assert lua.eval('calls') == 0


def test_camera_loss_freezes_timing_then_waits_for_ai_cleanup_and_receipt_before_slow_flush():
    lua = timed_camera(motion=True)
    lua.execute('''
      script.update(0) -- activate existing finite teacher lease
      assert(car.isAIControlled and not sim.isPaused)
      local save=io.save
      flushAttempts=0
      io.save=function(path,value,ensure)
        if path=='mock-camera/source-timing-v1.json' then
          flushAttempts=flushAttempts+1
          assert(not car.isAIControlled and sim.isPaused)
          assert(files[terminal]~=nil) -- immutable lifecycle receipt committed first
          now=now+1 -- injected expensive terminal persistence must be after cleanup
        end
        return save(path,value,ensure)
      end
      now=1.2; camera.last_capture_clock=now; files['mock-camera/stop']=true
      script.update(0) -- camera closes after this update's teacher observer
      assert(car.isAIControlled and flushAttempts==0)
      assert(ledger.records_frozen); frozenCount=#ledger.records
      now=1.21; script.update(0) -- camera loss disables AI, no timing disk write
      assert(not car.isAIControlled and flushAttempts==0)
      now=1.22; script.update(0) -- observe brake hold and pause
      assert(sim.isPaused and flushAttempts==0)
      failures[terminal]=1
      now=1.23; script.update(0) -- receipt write still pending
      assert(files[terminal]==nil and flushAttempts==0)
      now=1.5; script.update(0) -- same receipt recovery, then sole timing flush
    ''')
    assert lua.eval('flushAttempts') == 1
    assert lua.eval('#ledger.records == frozenCount')
    assert lua.eval('ledger.quiescence.pending_session_receipt_inactive')
    assert lua.eval("files[terminal].status") == 'api_returned_false'
    assert lua.eval("ledger.quiescence.observed_clock < ledger.finished_clock+1e-10")


def test_app_release_skips_timing_write_when_cleanup_cannot_be_observed_quiescent():
    lua = timed_camera(motion=True)
    lua.execute('script.update(0); releaseCallback()')
    assert lua.eval('car.isAIControlled') is False
    assert lua.eval("files['mock-camera/source-timing-v1.json']") is None
    assert lua.eval('ledger.records_frozen')


@pytest.mark.parametrize('change', ["request.source_timing_v1=1",
                                   'request.mmf_metadata=false',
                                   'request.deadline_seconds=61'])
def test_opt_in_rejects_wrong_schema_or_unbounded_diagnostic_before_renderer(change):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      request={id='mmf-stream-'..string.rep('a',32),frames=350,
        mmf_metadata=true,shared_transport=true,image_format='bmp',
        host_persistence_protocol='immutable_ram_v3',period_seconds=.04,
        deadline_seconds=19,exposure=.35,source_timing_v1=true}
      files['mock/flybrain/csp-camera/request.json']=request
      ac.findNodes=function() rendererEntered=true; error('renderer entered') end
    ''')
    lua.execute(change+'; script.update(0)')
    assert not lua.eval('rendererEntered')
    assert 'Source timing' in lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('option,mmf', [(1, False), (True, False)])
def test_host_rejects_source_timing_before_any_native_access(option, mmf):
    with pytest.raises(ValueError, match='Source timing'):
        CSPVisionRecorder(0, {}, source_timing_v1=option, mmf_metadata=mmf)


def fixture_ledger():
    return dict(schema='flybrain.csp.source_timing.v1', request_id='one', generation='g',
                clock_domain='csp_os_preciseClock_seconds', active_capture_closed=True,
                records_frozen=True, flushed_after_capture=True, actor_input=False,
                timing_observation_only=True,
                started_clock=1., capture_closed_clock=1.182, frozen_clock=1.182,
                finished_clock=1.2, final_capture_count=1, update_index=1,
                quiescence=dict(observed_clock=1.2, paused=True, ai_controlled=False,
                    speed_kmh=0., gas=0., brake=1., camera_inactive=True,
                    expert_probe_inactive=True, brake_settle_inactive=True,
                    pending_session_receipt_inactive=True),
                per_frame_timing_disk_writes=0, max_records=131072, overflow_records=0,
                record_fields=['stage', 'started_clock', 'finished_clock', 'duration_seconds',
                               'succeeded', 'update_index', 'capture_index', 'detail'],
                record_count=1, records=[['pollSession', 1., 1.182, .182, True, 1, 0, False]])


@pytest.mark.parametrize('mutation', [dict(generation='other'), dict(active_capture_closed=False),
    dict(record_count=2), dict(actor_input=True), dict(max_records=131073),
    dict(per_frame_timing_disk_writes=1)])
def test_terminal_proof_rejects_identity_open_capture_and_changed_contract(mutation):
    value = fixture_ledger()
    value.update(mutation)
    with pytest.raises(ValueError, match='Exact closed'):
        validate_source_timing_ledger(value, 'one', 'g')


def test_terminal_proof_keeps_original_clocks_and_honestly_reports_overflow():
    value = fixture_ledger()
    before = copy.deepcopy(value)
    assert validate_source_timing_ledger(value, 'one', 'g')['status'] == 'complete'
    assert value == before
    value['overflow_records'] = 3
    assert validate_source_timing_ledger(value, 'one', 'g')['status'] == 'overflow'
    value['records'][0][2] = .5
    with pytest.raises(ValueError, match='clock/duration'):
        validate_source_timing_ledger(value, 'one', 'g')

def test_pinned_long_recording_requires_explicit1600_scope_and_keeps_all_original_clocks():
    value=fixture_ledger();value['final_capture_count']=1566;before=copy.deepcopy(value)
    with pytest.raises(ValueError,match='boundary'):validate_source_timing_ledger(value,'one','g',expected_capture_count=1566)
    assert validate_source_timing_ledger(value,'one','g',expected_capture_count=1566,maximum_capture_count=1600)['status']=='complete'
    assert value==before
    value['final_capture_count']=1601
    with pytest.raises(ValueError,match='boundary'):validate_source_timing_ledger(value,'one','g',maximum_capture_count=1600)


@pytest.mark.parametrize('field,new_value', [('capture_closed_clock', .5),
    ('frozen_clock', 1.1), ('finished_clock', 1.19), ('final_capture_count', 2)])
def test_original_boundary_or_terminal_frame_disagreement_cannot_qualify(field, new_value):
    value = fixture_ledger()
    value[field] = new_value
    with pytest.raises(ValueError, match='boundary|clock/duration'):
        validate_source_timing_ledger(value, 'one', 'g', expected_capture_count=1)


def test_nested_call_completion_order_is_preserved_without_sorting_or_retiming():
    value = fixture_ledger()
    value['records'].append(['sessionState', 1.01, 1.1, .09, True, 1, 0, False])
    value['record_count'] = 2
    before = copy.deepcopy(value)
    assert validate_source_timing_ledger(value, 'one', 'g')['status'] == 'complete'
    assert value == before


@pytest.mark.parametrize('overflow', [0, 1])
def test_host_terminal_proof_retains_original_ledger_hash_and_metadata_archive(
        tmp_path, monkeypatch, overflow):
    from flybrain_drive import ac_csp_vision as vision
    from flybrain_drive.graph import sha256

    producer, local = tmp_path/'producer', tmp_path/'consumer'
    producer.mkdir()
    local.mkdir()
    (producer/'result.json').write_text(json.dumps(dict(status='completed', frames=1)))
    ledger_path = producer/'source-timing-v1.json'
    ledger = fixture_ledger()
    ledger['overflow_records'] = overflow
    ledger_path.write_text(json.dumps(ledger))
    original_hash = sha256(ledger_path)
    recorder = object.__new__(CSPVisionRecorder)
    recorder.stop = SimpleNamespace(set=lambda: None)
    recorder.thread = SimpleNamespace(join=lambda **kwargs: None, is_alive=lambda: False)
    recorder.triggered = recorder.source_timing_v1 = True
    recorder.source_body_recording=False
    recorder.request_id, recorder.directory, recorder.local = 'one', producer, local
    recorder.error = None
    recorder.count, recorder.bytes = 1, 100
    recorder.root, recorder.window = tmp_path, SimpleNamespace(hwnd=0)
    recorder.host_clock = lambda: 1.
    recorder.clock = SimpleNamespace(generation='g', record=lambda: dict(uncertainty_seconds=.002))
    monkeypatch.setattr(vision, 'clock_ping', lambda *args, **kwargs: None)
    result = recorder.close()
    assert result['source_timing']['status'] == ('overflow' if overflow else 'complete')
    assert result['status'] == ('failed' if overflow else 'stopped')
    assert result['source_timing']['sha256'] == original_hash == sha256(ledger_path)
    proof = json.loads((local/'source-timing-proof.json').read_text())
    assert proof == result['source_timing']
    recorder.archive_to(tmp_path/'archive')
    assert sha256(tmp_path/'archive/source-timing-v1.json') == original_hash
    assert (tmp_path/'archive/source-timing-proof.json').is_file()
