"""Execute the deployed Lua control flow with injected filesystem/API failures, no game."""

import faulthandler
import sys
from pathlib import Path

import pytest

lua_module = pytest.importorskip('lupa.luajit21')


@pytest.fixture(autouse=True)
def handled_luajit_errors_are_not_python_fatal_signals():
    # Injected errors are caught by Lua pcall, but LuaJIT's Windows SEH trips faulthandler.
    restore = sys.platform == 'win32' and faulthandler.is_enabled()
    if restore:
        faulthandler.disable()
    try:
        yield
    finally:
        if restore:
            faulthandler.enable()


def service(source_path=None):
    lua = lua_module.LuaRuntime(unpack_returned_tuples=True)
    lua.execute('''
      files, failures, calls, now = {}, {}, 0, 1
      sim = {windowHandle=17, isOnlineRace=false, isInMainMenu=false,
             isPaused=false, isLive=true, frame=0, idealLineShown=false}
      car = {speedKmh=0, gas=0, brake=1}
      script, ui = {}, {}
      os.preciseClock = function() return now end
      io.createDir = function() end
      io.load = function(path) return files[path] end
      io.fileExists = function(path) return files[path] ~= nil end
      io.save = function(path, value)
        if (failures[path] or 0) > 0 then
          failures[path] = failures[path] - 1
          return false
        end
        files[path] = value
        return true
      end
      JSON = {parse=function(v) return v end, stringify=function(v) return v end}
      ac = {FolderID={AppDataLocal=1}, getFolder=function() return 'mock' end,
            getSim=function() return sim end, getCar=function() return car end,
            getCarID=function() return 'bmw_z4_gt3' end,
            getTrackID=function() return 'monza' end,
            getPatchVersion=function() return 'mock' end,
            getPatchVersionCode=function() return 1 end,
            onSessionStart=function() end, onRelease=function(f) releaseCallback=f end}
      ac.tryToPause = function(value)
        calls = calls + 1
        if apiThrow then error('unknown API outcome') end
        sim.isPaused = value
        sim.isLive = not value
        return true
      end
      ac.tryToRestartSession = function() calls=calls+1; return true end
      ac.tryToStart = function() calls=calls+1; return true end
    ''')
    path = Path(source_path or 'integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    source = path.read_text(encoding='utf8')
    lua.execute(source)
    lua.execute('''
      files['mock/flybrain/csp-session/request.json'] = {
        schema='flybrain.csp.session.request.v1', id=string.rep('a',32), command='pause',
        generation='17:1', epoch=0, expires_clock=4}
    ''')
    return lua, source


def test_old_generation_terminal_camera_request_is_not_replayed_after_reload():
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='previous',
        expected_generation='old'}
      files['mock/flybrain/csp-camera/previous/result.json']={status='stopped', frames=248}
      script.update(0); now=1.3; script.update(0)
    ''')
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].state") == 'idle'
    assert lua.eval("files['mock/flybrain/csp-camera/previous/result.json'].frames") == 248
    assert lua.eval('calls') == 0


def test_old_generation_camera_request_without_terminal_still_fails_closed():
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='unfinished',
        expected_generation='old', frames=1, period_seconds=.1,
        deadline_seconds=3, exposure=.35}
      script.update(0)
    ''')
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].state") == 'failed'
    assert 'generation changed' in lua.eval(
        "files['mock/flybrain/csp-camera/status.json'].reason")
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('mutation', ['request.frames=9', 'sim.isPaused=false',
                                     'request.shared_transport=false', 'car.isAIControlled=true'])
def test_mmf_initial_qualification_cannot_activate_live_or_unbounded_transport(mutation):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      request={id='mmf-probe-'..string.rep('a',32), frames=8,
        mmf_metadata=true, shared_transport=true, image_format='bmp',
        period_seconds=.08, deadline_seconds=10, exposure=.35}
      files['mock/flybrain/csp-camera/request.json']=request
      mmfCalls=0
      ac.writeMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('should not open') end
      ac.readMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('should not open') end
    ''')
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('mmfCalls') == 0 and lua.eval('calls') == 0
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].state") == 'failed'
    assert 'eight paused' in lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")


@pytest.mark.parametrize('mutation', ['request.frames=1201',
    'request.host_persistence_protocol=nil', 'sim.isPaused=false',
    'request.verify_png_reference=true'])
def test_mmf_stream_requires_explicit_v3_budget_and_initial_stationary_state(mutation):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      request={id='mmf-stream-'..string.rep('a',32), frames=1200,
        mmf_metadata=true, shared_transport=true, image_format='bmp',
        host_persistence_protocol='immutable_ram_v3',
        period_seconds=.04, deadline_seconds=53, exposure=.35}
      files['mock/flybrain/csp-camera/request.json']=request
      mmfCalls=0
      ac.writeMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('should not open') end
      ac.readMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('should not open') end
    ''')
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('mmfCalls') == 0 and lua.eval('calls') == 0
    assert 'bounded v3 stream' in lua.eval(
        "files['mock/flybrain/csp-camera/status.json'].reason")


def test_engine_restart_commits_intent_once_and_never_claims_void_api_success():
    lua, source = service()
    lua.execute('''
      sim.isPaused=true; car.isAIControlled=false
      local request=files['mock/flybrain/csp-session/request.json']
      request.command='restart-engine'; request.old_engine_pid=42
      request.released_controller_verified=true
      ac.restartAssettoCorsa=function(value)
        assert(value==nil); calls=calls+1
        assert(files['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']
          .status=='engine_restart_intent_committed')
      end
      script.update(0); now=1.3; script.update(0)
    ''')
    assert lua.eval('calls') == 1
    assert lua.eval("files['mock/flybrain/csp-session/'..string.rep('a',32)..'.engine-call.json']"
                    ".status") == 'void_api_returned'
    lua.execute(source)
    lua.execute('now=2; script.update(0)')
    assert lua.eval('calls') == 1


@pytest.mark.parametrize('unsafe', [
    'sim.isPaused=false', 'car.isAIControlled=true', 'car.speedKmh=1',
    'request.released_controller_verified=false', 'request.old_engine_pid=0',
    "request.race_ini='unreviewed configuration'",
])
def test_engine_restart_rejects_changed_ownership_motion_and_arbitrary_ini(unsafe):
    lua, _ = service()
    lua.execute('''
      sim.isPaused=true; car.isAIControlled=false
      request=files['mock/flybrain/csp-session/request.json']
      request.command='restart-engine'; request.old_engine_pid=42
      request.released_controller_verified=true
      ac.restartAssettoCorsa=function() calls=calls+1 end
    ''')
    lua.execute(unsafe)
    lua.execute('script.update(0)')
    assert lua.eval('calls') == 0


def test_engine_restart_without_durable_intent_never_replays():
    lua, source = service()
    lua.execute('''
      sim.isPaused=true; car.isAIControlled=false
      request=files['mock/flybrain/csp-session/request.json']
      request.command='restart-engine'; request.old_engine_pid=42
      request.released_controller_verified=true
      ac.restartAssettoCorsa=function() calls=calls+1; error('unknown restart outcome') end
      failures['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']=1
      script.update(0); now=1.3; script.update(0)
    ''')
    assert lua.eval('calls') == 0
    lua.execute(source)
    lua.execute('now=2; script.update(0)')
    assert lua.eval('calls') == 0


def test_engine_restart_unknown_void_api_outcome_remains_non_replayable():
    lua, source = service()
    lua.execute('''
      sim.isPaused=true; car.isAIControlled=false
      request=files['mock/flybrain/csp-session/request.json']
      request.command='restart-engine'; request.old_engine_pid=42
      request.released_controller_verified=true
      ac.restartAssettoCorsa=function() calls=calls+1; error('unknown restart outcome') end
      script.update(0); now=1.3; script.update(0)
    ''')
    assert lua.eval('calls') == 1
    assert lua.eval("files['mock/flybrain/csp-session/'..string.rep('a',32)..'.engine-call.json']"
                    ".status") == 'api_outcome_unknown'
    lua.execute(source)
    lua.execute('now=2; script.update(0)')
    assert lua.eval('calls') == 1


def wind_candidate_service(tmp_path):
    from scripts.build_wind_restart_lua_candidate import build_candidate

    original = Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua').read_text(
        encoding='utf8')
    diagnostic = Path('integrations/ac_csp/candidates/RelativeAirflowDiagnostic.lua').read_text(
        encoding='utf8')
    path = tmp_path/'candidate.lua'
    path.write_text(build_candidate(original, diagnostic), encoding='utf8')
    lua, source = service(path)
    lua.execute('''
      sim.isPaused=true; car.isAIControlled=false
      request=files['mock/flybrain/csp-session/request.json']
      request.command='restart-engine-wind-diagnostic'; request.old_engine_pid=42
      request.released_controller_verified=true; request.native_ai_off_verified=true
      request.original_race_sha256=string.rep('a',64)
      request.wind_direction_deg=90; request.wind_speed_kmh=36
      ac.restartAssettoCorsa=function(value)
        assert(files['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']
          .status=='engine_restart_intent_committed')
        captured=value; calls=calls+1
      end
    ''')
    return lua, source


def test_wind_candidate_commits_before_wind_only_merge_and_never_replays(tmp_path):
    lua, source = wind_candidate_service(tmp_path)
    lua.execute('script.update(0); now=1.3; script.update(0)')
    assert lua.eval('calls') == 1
    assert lua.eval('captured') == ('[WIND]\nDIRECTION_DEG=90\n'
                                    'SPEED_KMH_MIN=36\nSPEED_KMH_MAX=36\n')
    assert lua.eval("files['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']"
                    '.wind_direction_deg') == 90
    assert lua.eval("files['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']"
                    '.wind_speed_kmh') == 36
    lua.execute(source)
    lua.execute('now=2; script.update(0)')
    assert lua.eval('calls') == 1


@pytest.mark.parametrize('unsafe', [
    'sim.isPaused=false', 'car.isAIControlled=true', 'request.native_ai_off_verified=false',
    'request.released_controller_verified=false', 'request.wind_direction_deg=91',
    'request.wind_speed_kmh=37', 'request.wind_speed_kmh=0; request.wind_direction_deg=90',
    "request.original_race_sha256='bad'", "request.race_ini='[RACE]\\nCARS=99'",
])
def test_wind_candidate_rejects_unqualified_or_unbounded_restart(tmp_path, unsafe):
    lua, _ = wind_candidate_service(tmp_path)
    lua.execute(unsafe)
    lua.execute('script.update(0)')
    assert lua.eval('calls') == 0


def test_wind_candidate_failed_terminal_commit_never_calls_engine(tmp_path):
    lua, _ = wind_candidate_service(tmp_path)
    lua.execute("failures['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']=1")
    lua.execute('script.update(0); now=1.3; script.update(0)')
    assert lua.eval('calls') == 0


def test_wind_observation_is_diagnostic_only_without_control_calls(tmp_path):
    lua, _ = wind_candidate_service(tmp_path)
    lua.execute('''
      request.command='wind-observe-diagnostic'
      sim.windVelocityKmh={x=0,y=36}; sim.windSpeedKmh=36
      car.velocity={x=0,y=0,z=0}; car.look={x=0,y=0,z=1}
      car.side={x=1,y=0,z=0}; car.up={x=0,y=1,z=0}; car.timestamp=42
      ac.getWindVelocity=function() return {x=0,y=0,z=10} end
      script.update(0)
    ''')
    assert lua.eval('calls') == 0
    terminal = "files['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']"
    assert lua.eval(terminal+'.status') == 'wind_observed_diagnostic'
    assert lua.eval(terminal+'.actor_admissible') is False
    assert lua.eval(terminal+'.observation.source_verified') is False
    assert lua.eval(terminal+'.observation.wind_velocity_candidate_m_s[3]') == 10


@pytest.mark.parametrize('unsafe', ['sim.isPaused=false', 'car.isAIControlled=true',
                                    'car.speedKmh=1'])
def test_wind_observation_rejects_live_or_unowned_state(tmp_path, unsafe):
    lua, _ = wind_candidate_service(tmp_path)
    lua.execute("request.command='wind-observe-diagnostic'; " + unsafe)
    lua.execute('script.update(0)')
    assert lua.eval('calls') == 0
    assert lua.eval("files['mock/flybrain/csp-session/'..string.rep('a',32)..'.json'].status") == (
        'rejected')


def test_disable_ideal_line_is_observed_idempotent_sim_api_without_os_input():
    lua, _ = service()
    lua.execute('''
      sim.isPaused=true; sim.idealLineShown=true; car.isAIControlled=false
      ac.trySimKeyPressCommand=function(command)
        assert(command=='Ideal Line'); calls=calls+1; sim.idealLineShown=false; return true
      end
      files['mock/flybrain/csp-session/request.json'].command='disable-ideal-line'
      script.update(0)
    ''')
    assert lua.eval('calls') == 1
    assert lua.eval('sim.idealLineShown') is False
    lua.execute('''
      now=now+1
      files['mock/flybrain/csp-session/request.json'].id=string.rep('b',32)
      script.update(0)
    ''')
    assert lua.eval('calls') == 1


def test_disable_line_rejects_unknown_state_without_toggle():
    lua, _ = service()
    lua.execute('''
      sim.isPaused=true; sim.idealLineShown=nil; car.isAIControlled=false
      ac.trySimKeyPressCommand=function() calls=calls+1; return true end
      files['mock/flybrain/csp-session/request.json'].command='disable-ideal-line'
      script.update(0)
    ''')
    assert lua.eval('calls') == 0
    status = lua.eval("files['mock/flybrain/csp-session/'..string.rep('a',32)..'.json'].status")
    assert status == 'rejected'


@pytest.mark.parametrize('state,diagnostic,frames,allowed', [
    ('true', False, 4, False), ('nil', False, 4, False), ('false', False, 4, True),
    ('true', True, 4, True), ('true', True, 9, False)])
def test_line_off_capture_and_bounded_diagnostic(state, diagnostic, frames, allowed):
    lua, _ = service()
    lua.execute("files['mock/flybrain/csp-session/request.json']=nil; sim.isPaused=true")
    lua.execute('sim.idealLineShown='+state)
    lua.execute("ac.findNodes=function() rendererEntered=true; error('mock renderer entry') end")
    request = lua.table_from(dict(id='aid-probe', frames=frames, period_seconds=.2,
        deadline_seconds=10, exposure=.35, expected_generation='17:1',
        allow_ideal_line_probe=diagnostic))
    lua.globals().files['mock/flybrain/csp-camera/request.json'] = request
    lua.execute('script.update(0)')
    assert bool(lua.eval('rendererEntered')) == allowed


def test_terminal_write_retry_never_repeats_api_or_overwrites_accepted():
    lua, _ = service()
    lua.execute('''
      terminal = 'mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json'
      accepted = 'mock/flybrain/csp-session/'..string.rep('a',32)..'.json'
      failures[terminal] = 2
      script.update(0)
    ''')
    assert lua.eval('calls') == 1
    assert lua.eval('files[accepted].status') == 'accepted'
    assert lua.eval('files[terminal]') is None
    lua.execute('now=1.3; script.update(0)')
    assert lua.eval("files['mock/flybrain/csp-session/status.json'].error")
    lua.execute('now=1.6; script.update(0)')
    assert lua.eval('calls') == 1
    assert lua.eval('files[accepted].status') == 'accepted'
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval("files['mock/flybrain/csp-session/status.json'].error") is None


def test_transient_heartbeat_failure_recovers_without_poisoning_command_service():
    lua, _ = service()
    lua.execute("failures['mock/flybrain/csp-session/status.json']=1; script.update(0)")
    lua.execute('now=1.3; script.update(0)')
    assert lua.eval('calls') == 1
    assert lua.eval("files['mock/flybrain/csp-session/status.json'].error") is None
    assert lua.eval("files['mock/flybrain/csp-session/status.json'].heartbeat_write_failures") == 1


def test_unknown_api_outcome_and_app_reload_never_replay_accepted_request():
    lua, source = service()
    lua.execute('apiThrow=true; script.update(0); now=1.3; script.update(0)')
    assert lua.eval('calls') == 1
    assert lua.eval("files['mock/flybrain/csp-session/status.json'].error")
    lua.execute('now=2; apiThrow=false')
    lua.execute(source)
    lua.execute('script.update(0)')
    assert lua.eval('calls') == 1


def test_nonstationary_request_rejected_without_side_effect():
    lua, _ = service()
    lua.execute('car.speedKmh=5; script.update(0)')
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('allowed,ai', [(False, False), (True, True)])
def test_readonly_capability_getters_preserve_false_and_never_call_autopilot(allowed, ai):
    lua, _ = service()
    lua.execute("files['mock/flybrain/csp-session/request.json']=nil; aiCalls=0")
    lua.execute('physics={allowed=function() return '+str(allowed).lower()+
        ' end, setCarAutopilot=function() aiCalls=aiCalls+1 end}')
    lua.globals().car.isAIControlled = ai
    lua.execute('script.update(0)')
    status = lua.globals().files['mock/flybrain/csp-session/status.json']
    assert status.physics_api_allowed is allowed
    assert status.ai_controlled is ai
    assert lua.eval('aiCalls') == 0 and lua.eval('calls') == 0


def test_unsupported_capability_getters_stay_unknown_without_poisoning_lifecycle():
    lua, _ = service()
    lua.execute('physics={allowed=function() error("getter unavailable") end}')
    lua.execute('script.update(0)')
    status = lua.globals().files['mock/flybrain/csp-session/status.json']
    assert status.physics_api_allowed is None and status.ai_controlled is None
    assert status.error is None and lua.eval('calls') == 1


@pytest.mark.parametrize('frames,enters_renderer',
                         [(1650, True), (3250, True), (4001, False)])
def test_extended_camera_request_budget_is_checked_by_actual_lua(frames, enters_renderer):
    lua, _ = service()
    lua.execute("files['mock/flybrain/csp-session/request.json']=nil")
    lua.execute("ac.findNodes=function() rendererEntered=true; error('mock renderer entry') end")
    request = lua.table_from(dict(id='camera-budget-test', frames=frames, period_seconds=.03,
                                 deadline_seconds=102.5, exposure=.35, expected_generation='17:1'))
    lua.globals().files['mock/flybrain/csp-camera/request.json'] = request
    lua.execute('script.update(0)')
    assert bool(lua.eval('rendererEntered')) == enters_renderer
    assert lua.eval('calls') == 0


def probe_service():
    lua, source = service()
    lua.execute('''
      sim.isPaused=true; sim.isLive=false
      car.isAIControlled=false; car.position={x=0,y=0,z=0}; setters={}
      physics={allowed=function() return true end,
        setAITopSpeed=function(index,limit)
          table.insert(setters, 'limit:'..tostring(limit))
        end,
        setCarAutopilot=function(enabled,chooseTyres)
          table.insert(setters, enabled and 'enable' or 'disable')
          car.isAIControlled=enabled
          if enabled and activationThrow then error('after enable') end
        end}
      request=files['mock/flybrain/csp-session/request.json']
      request.command='expert-probe'; request.speed_limit_kmh=15; request.hold_seconds=.5
      terminal='mock/flybrain/csp-session/'..request.id..'.terminal.json'
      accepted='mock/flybrain/csp-session/'..request.id..'.json'
    ''')
    return lua, source


def brake_settle_service():
    lua, source = probe_service()
    lua.execute('''
      request.command='brake-settle-after-teacher'
      request.terminal_teacher_request_id=string.rep('b',32)
      request.native_residual_after_teacher_verified=true
      request.native_residual_speed_kmh=.108073726
      car.speedKmh=.045879837
      previous={status='api_returned_true',command='expert-collect30-speed30',
        after={window_handle='17',epoch=0,paused=true,ai_controlled=false,
          car_id='bmw_z4_gt3',track_id='monza'}}
      files['mock/flybrain/csp-session/'..string.rep('b',32)..'.terminal.json']=previous
    ''')
    return lua, source


def test_brake_settle_runs_without_client_or_control_setters_and_never_replays():
    lua, source = brake_settle_service()
    lua.execute('script.update(0); now=1.05; car.speedKmh=.005; script.update(0)')
    assert lua.eval('sim.isPaused') is False
    lua.execute('now=1.13; script.update(0)')
    assert lua.eval('sim.isPaused') is True
    assert lua.eval('files[terminal].brake_settle_passed') is True
    assert lua.eval('#setters') == 0
    assert lua.eval('calls') == 2
    lua.execute('now=2')
    lua.execute(source)
    lua.execute('script.update(0)')
    assert lua.eval('calls') == 2


def test_brake_settle_waits_for_asynchronous_pause_without_repeating_setter():
    lua, _ = brake_settle_service()
    lua.execute('''
      script.update(0)
      ac.tryToPause=function(value) calls=calls+1; return true end
      now=1.13; car.speedKmh=.005; script.update(0)
    ''')
    assert lua.eval('files[terminal]') is None
    lua.execute('now=1.14; script.update(0)')
    assert lua.eval('calls') == 2
    lua.execute('sim.isPaused=true; now=1.15; script.update(0)')
    assert lua.eval('files[terminal].brake_settle_passed') is True
    assert lua.eval('calls') == 2


@pytest.mark.parametrize('unsafe', [
    'previous.status="api_outcome_unknown"', 'previous.after.window_handle="18"',
    'previous.after.epoch=1', 'previous.after.ai_controlled=true',
    'request.native_residual_after_teacher_verified=false',
    'request.native_residual_speed_kmh=.2', 'request.native_residual_speed_kmh=.099',
    'request.terminal_teacher_request_id="../../bad"', 'request.expires_clock=0',
    'car.speedKmh=.1', 'car.brake=.98', 'car.gas=.02', 'car.isAIControlled=true',
    'sim.isOnlineRace=true', 'sim.isPaused=false',
])
def test_brake_settle_rejects_unverified_source_or_unsafe_state(unsafe):
    lua, _ = brake_settle_service()
    lua.execute(unsafe+'; script.update(0)')
    assert lua.eval('calls') == 0
    assert lua.eval('files[accepted].status') == 'rejected'


@pytest.mark.parametrize('violation', [
    'car.position.x=.021', 'car.gas=.03', 'car.brake=.97', 'car.speedKmh=.2',
    'car.isAIControlled=true', 'car.position=nil',
])
def test_brake_settle_pauses_on_envelope_change(violation):
    lua, _ = brake_settle_service()
    lua.execute('script.update(0); now=1.02; '+violation+'; script.update(0)')
    assert lua.eval('sim.isPaused') is True
    assert lua.eval('files[terminal].brake_settle_passed') is False
    assert lua.eval('#setters') == 0


def test_brake_settle_deadline_and_app_release_pause_without_reactivation():
    lua, _ = brake_settle_service()
    lua.execute('script.update(0); now=1.36; script.update(0)')
    assert lua.eval('sim.isPaused') is True
    assert lua.eval('files[terminal].brake_settle_passed') is False
    released, _ = brake_settle_service()
    released.execute('script.update(0); releaseCallback()')
    assert released.eval('sim.isPaused') is True
    assert released.eval('#setters') == 0


def test_stationary_probe_is_owned_once_and_expires_without_client():
    lua, _ = probe_service()
    lua.execute('script.update(0)')
    assert lua.eval('table.concat(setters,",")') == 'limit:15,enable'
    assert lua.eval('files[accepted].status') == 'accepted'
    lua.execute('now=1.1; script.update(0); now=1.6; script.update(0)')
    assert lua.eval('table.concat(setters,",")') == 'limit:15,enable,disable'
    assert lua.eval('files[terminal]') is None
    lua.execute('now=1.7; script.update(0); now=2; script.update(0)')
    assert lua.eval('#setters') == 4
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[terminal].probe_seen_enabled') is True
    assert lua.eval('files[terminal].maximum_position_delta_m') == 0
    assert lua.eval('files[terminal].driving_tested') is False
    assert lua.eval('sim.isPaused') is True and lua.eval('car.isAIControlled') is False
    assert lua.eval('calls') == 0  # no pause/resume or vJoy driving


def test_probe_terminal_write_failure_retries_metadata_only():
    lua, _ = probe_service()
    lua.execute('''
      failures[terminal]=2; script.update(0); now=1.1; script.update(0)
      now=1.6; script.update(0); now=1.7; script.update(0)
      now=1.8; script.update(0); now=2; script.update(0)
    ''')
    assert lua.eval('#setters') == 4
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[accepted].status') == 'accepted'


def test_probe_activation_unknown_is_disabled_and_never_replayed_on_reload():
    lua, source = probe_service()
    lua.execute('activationThrow=true; script.update(0); now=1.1; script.update(0)')
    assert lua.eval('car.isAIControlled') is False
    lua.execute('now=1.2; script.update(0)')
    assert lua.eval('files[terminal].status') == 'api_outcome_unknown'
    assert lua.eval('#setters') == 4
    lua.execute('now=2')
    lua.execute(source)
    lua.execute('script.update(0)')
    assert lua.eval('#setters') == 4


@pytest.mark.parametrize('mutation', [
    'sim.isPaused=false', 'car.speedKmh=1',
    'physics.allowed=function() return false end', 'car.isAIControlled=nil',
    'car.position.x=0/0', 'car.position=nil', 'request.hold_seconds=10',
    'request.speed_limit_kmh=50', 'sim.isOnlineRace=true',
])
def test_probe_invalid_preconditions_reject_without_setters(mutation):
    lua, _ = probe_service()
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0
    assert lua.eval('files[accepted].status') == 'rejected'


@pytest.mark.parametrize('mutation', ['car.position.x=.03', 'car.position=nil',
                                      'sim.isPaused=false'])
def test_probe_stationarity_violation_cannot_pass(mutation):
    lua, _ = probe_service()
    lua.execute('script.update(0); now=1.1; '+mutation+'; script.update(0)')
    lua.execute('now=1.6; script.update(0); now=1.7; script.update(0)')
    assert lua.eval('car.isAIControlled') is False
    assert lua.eval('files[terminal].status') == 'api_returned_false'


def test_probe_blocks_new_mutations_during_lease():
    lua, _ = probe_service()
    lua.execute('''
      script.update(0)
      request.id=string.rep('b',32); request.command='resume'
      now=1.2; script.update(0)
    ''')
    assert lua.eval('calls') == 0 and lua.eval('sim.isPaused') is True


def test_probe_release_disables_ai_and_accepted_request_stays_nonreplayable():
    lua, source = probe_service()
    lua.execute('script.update(0); releaseCallback()')
    assert lua.eval('car.isAIControlled') is False
    assert lua.eval('table.concat(setters,",")') == 'limit:15,enable,disable'
    lua.execute('now=2')
    lua.execute(source)
    lua.execute('script.update(0)')
    assert lua.eval('#setters') == 3
    assert lua.eval('files[terminal]') is None  # release outcome remains incomplete


def test_probe_teardown_not_observed_reports_unknown_without_repeating_disable():
    lua, _ = probe_service()
    lua.execute('''
      script.update(0)
      physics.setCarAutopilot=function(enabled) table.insert(setters,'disable ignored') end
      now=1.1; script.update(0); now=1.6; script.update(0)
      now=3.2; script.update(0); now=4; script.update(0)
    ''')
    assert lua.eval('#setters') == 3
    assert lua.eval('files[terminal].status') == 'api_outcome_unknown'
    assert lua.eval("files['mock/flybrain/csp-session/status.json'].error")


def motion_service():
    lua, source = probe_service()
    lua.execute('''
      request.command='expert-motion-probe'; request.hold_seconds=3
      car.wheelsOutside=0; car.steer=0
    ''')
    return lua, source


def test_motion_probe_resumes_ai_once_then_brakes_before_pausing():
    lua, _ = motion_service()
    lua.execute('script.update(0)')
    assert lua.eval('table.concat(setters,",")') == 'enable,limit:15'
    assert lua.eval('sim.isPaused') is False and lua.eval('calls') == 1
    lua.execute('''
      now=1.1; car.speedKmh=10; car.gas=.2; car.brake=0; car.position.x=1
      script.update(0); now=4.1; car.position.x=10; script.update(0)
    ''')
    assert lua.eval('car.isAIControlled') is False
    assert lua.eval('sim.isPaused') is False  # retain physics until brake hold is observed
    lua.execute('now=4.2; car.speedKmh=0; car.gas=0; car.brake=1; script.update(0)')
    assert lua.eval('sim.isPaused') is True
    lua.execute('now=4.3; script.update(0)')
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[terminal].command') == 'expert-motion-probe'
    assert lua.eval('files[terminal].motion_probe_passed') is True
    assert lua.eval('files[terminal].stationary_probe_passed') is False
    assert lua.eval('files[terminal].maximum_speed_kmh') == 10
    assert lua.eval('#files[terminal].telemetry') == 4
    assert lua.eval('calls') == 2 and lua.eval('#setters') == 4


@pytest.mark.parametrize('mutation', [
    'car.speedKmh=21', 'car.position.x=21', 'car.wheelsOutside=3',
    'car.steer=nil', 'car.isAIControlled=false', 'sim.isOnlineRace=true',
])
def test_motion_probe_envelope_violation_tears_down_and_cannot_pass(mutation):
    lua, _ = motion_service()
    lua.execute('script.update(0); now=1.1; '+mutation+'; script.update(0)')
    assert lua.eval('car.isAIControlled') is False
    lua.execute('now=1.2; car.speedKmh=0; script.update(0); now=1.3; script.update(0)')
    assert lua.eval('files[terminal].status') == 'api_returned_false'


def test_motion_no_movement_fails_even_when_ai_was_enabled():
    lua, _ = motion_service()
    lua.execute('''
      script.update(0); now=1.1; script.update(0); now=4.1; script.update(0)
      now=4.2; script.update(0); now=4.3; script.update(0)
    ''')
    assert lua.eval('files[terminal].motion_probe_passed') is False


def test_motion_no_brake_hold_causes_emergency_pause_and_unknown():
    lua, _ = motion_service()
    lua.execute('''
      script.update(0); now=1.1; car.speedKmh=10; car.position.x=1; script.update(0)
      now=4.1; script.update(0); now=8.2; script.update(0)
    ''')
    assert lua.eval('sim.isPaused') is True
    assert lua.eval('files[terminal].status') == 'api_outcome_unknown'


@pytest.mark.parametrize('mutation', [
    'car.wheelsOutside=1', 'car.wheelsOutside=nil', 'sim.isInMainMenu=true',
    'request.hold_seconds=30',
])
def test_motion_start_requires_track_and_exact_finite_contract(mutation):
    lua, _ = motion_service()
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0 and lua.eval('calls') == 0


def test_motion_limit_failure_after_enable_does_not_resume_and_cleans_up():
    lua, _ = motion_service()
    lua.execute('''
      physics.setAITopSpeed=function() error('uncertain limit') end
      script.update(0); now=1.1; script.update(0); now=1.2; script.update(0)
    ''')
    assert lua.eval('sim.isPaused') is True and lua.eval('calls') == 0
    assert lua.eval('car.isAIControlled') is False
    assert lua.eval('files[terminal].status') == 'api_outcome_unknown'


def set_owned_camera(lua):
    # Inject an already running producer via the Lua debug API; the request guards
    # and state machine remain the unmodified deployed source.
    lua.execute('''
      function setCamera(value)
        for i=1,30 do
          local name,fn=debug.getupvalue(script.update,i)
          if name=='pollSession' then
            for j=1,30 do
              local key=debug.getupvalue(fn,j)
              if key=='active' then debug.setupvalue(fn,j,value); return end
            end
          end
        end
        error('active camera upvalue not found')
      end
      camera={request={id='owned-camera'},deadline=100,last_capture_clock=1,
        directory='mock-camera',next=100,views={}}
      setCamera(camera); request.camera_request_id='owned-camera'
    ''')


def brake_pulse_service():
    lua, source = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-brake-pulse30'; request.hold_seconds=30
      request.speed_limit_kmh=30; request.ai_steer_multiplier=.9
      request.brake_pulse_seconds=.4; request.brake_pulse_after_seconds=3
      physics.setAISteerMultiplier=function() end
      pulses={}
      physics.setAIStopCounter=function(index,duration)
        assert(index==0); table.insert(pulses,duration)
        if pulseThrow and duration>0 then error('after setter') end
      end
    ''')
    return lua, source


@pytest.mark.parametrize('throw', [False, True])
def test_brake_pulse_is_issued_once_and_cleared_without_repeating_unknown_setter(throw):
    lua, _ = brake_pulse_service()
    lua.globals().pulseThrow = throw
    lua.execute('''
      script.update(0)
      now=3.9; camera.last_capture_clock=now; car.speedKmh=30
      car.brake=0; car.gas=.1; car.position.x=10; script.update(0)
      assert(#pulses==0)
      now=4.1; camera.last_capture_clock=now; script.update(0)
      assert(#pulses==1 and pulses[1]==.4)
      now=4.2; camera.last_capture_clock=now; car.brake=.8; car.speedKmh=27
      script.update(0)
    ''')
    if throw:
        lua.execute('now=4.3; car.speedKmh=0; car.gas=0; car.brake=1; script.update(0);'
                    'now=4.4; script.update(0)')
    else:
        lua.execute('now=31.1; camera.last_capture_clock=now; script.update(0);'
                    'now=31.2; car.speedKmh=0; car.gas=0; car.brake=1; script.update(0);'
                    'now=31.3; script.update(0)')
    assert lua.eval('#pulses') == 2
    assert lua.eval('pulses[2]') == 0
    assert lua.eval('files[terminal].brake_pulse_passed') is (not throw)


@pytest.mark.parametrize('mutation', [
    'request.brake_pulse_seconds=.5', 'request.brake_pulse_after_seconds=2',
    'physics.setAIStopCounter=nil',
])
def test_brake_pulse_rejects_changed_duration_or_missing_api(mutation):
    lua, _ = brake_pulse_service()
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0 and lua.eval('calls') == 0


def test_brake_pulse_requires_actual_brake_and_speed_drop_before_cleanup():
    lua, _ = brake_pulse_service()
    lua.execute('''
      script.update(0)
      now=4.1; camera.last_capture_clock=now; car.speedKmh=30
      car.brake=0; car.gas=.1; car.position.x=10; script.update(0)
      now=4.2; camera.last_capture_clock=now; car.speedKmh=25; script.update(0)
      now=31.1; camera.last_capture_clock=now; script.update(0)
      now=31.2; car.speedKmh=0; car.gas=0; car.brake=1; script.update(0)
      now=31.3; script.update(0)
    ''')
    assert lua.eval('files[terminal].brake_pulse_passed') is False
    assert lua.eval('files[terminal].status') == 'api_returned_false'


def moving_handoff_service():
    lua, source = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-moving-handoff3'; request.hold_seconds=3
      request.ai_steer_multiplier=.9; request.student_owner_verified=true
      request.student_owner_pid=123; sim.isPaused=false; sim.isLive=true
      car.speedKmh=17; car.gas=.07; car.brake=0
      physics.setAISteerMultiplier=function() end
    ''')
    return lua, source


def test_moving_handoff_preserves_motion_without_resume_and_then_brakes_and_pauses():
    lua, _ = moving_handoff_service()
    lua.execute('script.update(0)')
    assert lua.eval('car.isAIControlled') is True and lua.eval('calls') == 0
    assert lua.eval('table.concat(setters,",")') == 'enable,limit:15'
    lua.execute('''
      now=1.1; camera.last_capture_clock=now; car.position.x=1; script.update(0)
      now=4.1; camera.last_capture_clock=now; car.position.x=14; script.update(0)
      now=4.2; camera.last_capture_clock=now; car.speedKmh=0; car.gas=0; car.brake=1
      script.update(0); now=4.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].moving_handoff_passed') is True
    assert lua.eval('files[terminal].student_owner_pid') == 123
    assert lua.eval('files[terminal].before.speed_kmh') == 17
    assert lua.eval('car.isAIControlled') is False and lua.eval('sim.isPaused') is True
    assert lua.eval('calls') == 1  # final pause only


@pytest.mark.parametrize('mutation', [
    'car.speedKmh=19', 'car.speedKmh=0', 'car.wheelsOutside=1', 'sim.isPaused=true',
    'request.student_owner_verified=nil', 'request.student_owner_pid=0',
    'request.student_owner_pid=1.5', 'setCamera(nil)', 'camera.deadline=8',
    'request.ai_steer_multiplier=1', 'request.hold_seconds=30',
    'request.command="expert-motion-probe"'])
def test_moving_handoff_is_exact_owned_bounded_and_never_weakens_stationary_modes(mutation):
    lua, _ = moving_handoff_service()
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0


def test_moving_handoff_camera_loss_disables_once_without_new_activation():
    lua, _ = moving_handoff_service()
    lua.execute('''
      script.update(0); now=1.1; setCamera(nil); script.update(0)
      now=1.2; car.speedKmh=0; car.gas=0; car.brake=1; script.update(0)
      now=1.3; script.update(0)
    ''')
    assert lua.eval('files[terminal].status') == 'api_returned_false'
    assert lua.eval('files[terminal].stop_reason') == 'owned camera ended or became stale'
    assert lua.eval('car.isAIControlled') is False
    assert lua.eval('table.concat(setters,",")').count('enable') == 1


def test_moving_handoff30_is_a_separate_owned_lease_and_keeps_20kmh_guard():
    lua, _ = moving_handoff_service()
    lua.execute('''
      request.command='expert-moving-handoff30'; request.hold_seconds=30
      script.update(0); now=1.1; camera.last_capture_clock=now
      car.position.x=1; script.update(0)
      now=31.1; camera.last_capture_clock=now; car.position.x=130; script.update(0)
      now=31.2; camera.last_capture_clock=now; car.speedKmh=0; car.gas=0; car.brake=1
      script.update(0); now=31.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].moving_handoff_passed') is True
    assert lua.eval('files[terminal].maximum_displacement_guard_m') == 180
    assert lua.eval('files[terminal].motion_hard_speed_kmh') == 20
    assert lua.eval('calls') == 1
    bad, _ = moving_handoff_service()
    bad.execute('''
      request.command='expert-moving-handoff30'; request.hold_seconds=30
      camera.deadline=34; script.update(0)
    ''')
    assert bad.eval('#setters') == 0
    old, _ = moving_handoff_service()
    old.execute('request.hold_seconds=30; script.update(0)')
    assert old.eval('#setters') == 0


def test_motion_can_share_only_the_pinned_fresh_camera_and_stops_on_loss():
    lua, _ = motion_service()
    set_owned_camera(lua)
    lua.execute('script.update(0)')
    assert lua.eval('car.isAIControlled') is True
    lua.execute('now=1.1; setCamera(nil); script.update(0)')
    assert lua.eval('car.isAIControlled') is False
    lua.execute('now=1.2; script.update(0); now=1.3; script.update(0)')
    assert lua.eval('files[terminal].status') == 'api_returned_false'
    assert lua.eval('files[terminal].stop_reason') == 'owned camera ended or became stale'


@pytest.mark.parametrize('mutation', [
    "request.camera_request_id='other-camera'", 'request.camera_request_id=nil',
    'camera.last_capture_clock=.5', 'camera.deadline=8',
])
def test_camera_lease_identity_freshness_and_remaining_budget_required(mutation):
    lua, _ = motion_service()
    set_owned_camera(lua)
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0


def test_motion_owned_cancellation_disables_ai_without_second_activation():
    lua, _ = motion_service()
    lua.execute('''
      script.update(0)
      files['mock/flybrain/csp-session/'..request.id..'.stop']=true
      now=1.1; script.update(0); now=1.2; script.update(0); now=1.3; script.update(0)
    ''')
    assert lua.eval('car.isAIControlled') is False
    assert lua.eval('files[terminal].stop_reason') == 'owned client cancellation'
    assert lua.eval('files[terminal].motion_probe_passed') is False


@pytest.mark.parametrize('fmt,paired,frames,allowed', [
    ('bmp', True, 8, True), ('bmp', False, 100, True), ('png', False, 8, True),
    ('jpeg', False, 8, False), ('bmp', True, 9, False), ('png', True, 8, False),
    ('bmp', 'yes', 8, False),
])
def test_lossless_transport_and_paired_reference_budget(fmt, paired, frames, allowed):
    lua, _ = service()
    lua.execute("files['mock/flybrain/csp-session/request.json']=nil")
    lua.execute("ac.findNodes=function() rendererEntered=true; error('mock renderer entry') end")
    request = lua.table_from(dict(id='transport-test', frames=frames, period_seconds=.03,
        deadline_seconds=10, exposure=.35, expected_generation='17:1',
        image_format=fmt, verify_png_reference=paired))
    lua.globals().files['mock/flybrain/csp-camera/request.json'] = request
    lua.execute('script.update(0)')
    assert bool(lua.eval('rendererEntered')) == allowed
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('mutation,allowed', [('', True), ('probe.frames=9', False),
    ('sim.isPaused=false', False), ('car.isAIControlled=true', False),
    ('car.speedKmh=1', False), ('car.brake=0', False), ('sim.isOnlineRace=true', False),
    ("probe.async_readback_probe='yes'", False)])
def test_async_readback_probe_requires_a_finite_paused_brake_held_car(mutation, allowed):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      ac.findNodes=function() rendererEntered=true; error('mock renderer entry') end
      probe={id='async-probe',frames=2,period_seconds=.2,deadline_seconds=10,
        exposure=.35,expected_generation='17:1',async_readback_probe=true}
    ''')
    lua.execute(mutation+"\n files['mock/flybrain/csp-camera/request.json']=probe; "
                'script.update(0)')
    assert bool(lua.eval('rendererEntered')) == allowed
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('mutation,allowed', [('', True), ('probe.frames=2', False),
    ('sim.isPaused=false', False), ('probe.async_readback_probe=true', False)])
def test_shared_texture_probe_only_holds_one_paused_immutable_group(mutation, allowed):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      ac.findNodes=function() rendererEntered=true; error('mock renderer entry') end
      probe={id='shared-probe',frames=1,period_seconds=.2,deadline_seconds=10,
        exposure=.35,expected_generation='17:1',shared_texture_probe=true}
    ''')
    lua.execute(mutation+"\n files['mock/flybrain/csp-camera/request.json']=probe; "
                'script.update(0)')
    assert bool(lua.eval('rendererEntered')) == allowed
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('ack', [None, 'valid', 'wrong_generation', 'wrong_id', 'wrong_frame'])
def test_shared_transport_keeps_textures_immutable_until_exact_owned_ack(ack):
    lua, _ = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      camera.frames=1; camera.request.frames=2; camera.pending=true; camera.awaiting_ack=0
    ''')
    if ack:
        lua.execute("files['mock-camera/ack-0000.json']="
                    "{request_id='owned-camera',generation='17:1',capture_id=0}")
        if ack != 'valid':
            field, value = {'wrong_generation': ('generation', "'other'"),
                            'wrong_id': ('request_id', "'other'"),
                            'wrong_frame': ('capture_id', '1')}[ack]
            lua.execute(f"files['mock-camera/ack-0000.json'].{field}={value}")
    lua.execute('script.update(0)')
    if ack in (None, 'valid'):
        assert lua.eval('camera.pending') == (ack is None)
        assert lua.eval('camera.frames') == 1
    else:
        assert lua.eval("files['mock-camera/result.json'].status") == 'failed'
    assert lua.eval('calls') == 0


def test_shared_transport_releases_final_group_only_after_ack():
    lua, _ = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      camera.frames=1; camera.request.frames=1; camera.pending=true; camera.awaiting_ack=0
      script.update(0)
      assert(files['mock-camera/result.json']==nil)
      files['mock-camera/ack-0000.json']={request_id='owned-camera',generation='17:1',capture_id=0}
      now=1.1; script.update(0)
    ''')
    assert lua.eval("files['mock-camera/result.json'].status") == 'completed'
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('cancel', [False, True])
def test_async_probe_waits_for_all_views_and_never_commits_after_cancel(cancel):
    lua, _ = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      callbacks, disposed = {}, 0
      ac.ImageFormat={PNG=1}
      function upvalue(fn,key)
        for i=1,60 do
          local name,value=debug.getupvalue(fn,i)
          if name==key then return value end
        end
        error('upvalue missing: '..key)
      end
      rgbm=function() return {} end
      camera.frames=0; camera.request.frames=1; camera.request.period_seconds=.2
      camera.views={}
      local layout={}
      for _,id in ipairs({'front','left','right','rear'}) do
        local shot={update=function() end,save=function() end,
          accessData=function(self,callback) table.insert(callbacks,callback) end,
          dispose=function() disposed=disposed+1 end}
        table.insert(camera.views,{id=id,shot=shot}); layout[id]={0,0,56}
      end
      local capture=upvalue(script.update,'capture')
      upvalue(capture,'asyncProbe')(car,layout,now)
      assert(camera.pending and camera.frames==0 and #callbacks==4)
      capture(now) -- must return before reading car geometry while pending
      dataDisposed=0
      function cpuData()
        return {size=function() return {x=320,y=180} end,
          colorTo=function(self,c,x,y) c.r=x%256/255; c.g=y/255; c.b=.5 end,
          dispose=function() dataDisposed=dataDisposed+1 end}
      end
    ''')
    if cancel:
        lua.execute('releaseCallback()')
    lua.execute('callbacks[1](nil,cpuData()); callbacks[2](nil,cpuData())')
    assert lua.eval("files['mock-camera/frame-0000.json']") is None
    lua.execute('callbacks[3](nil,cpuData()); callbacks[4](nil,cpuData())')
    assert lua.eval('dataDisposed') == 4
    assert lua.eval('disposed') == 4
    assert (lua.eval("files['mock-camera/frame-0000.json'] ~= nil")) != cancel
    if not cancel:
        assert lua.eval("#files['mock-camera/0000-front.ppm']") == 172815
        assert lua.eval("files['mock-camera/result.json'].status") == 'completed'


def test_bounded_expert_steer_multiplier_is_applied_after_enable_and_restored():
    lua, _ = motion_service()
    lua.execute('''
      request.ai_steer_multiplier=.9
      physics.setAISteerMultiplier=function(index,value)
        table.insert(setters,'steer:'..value)
      end
      script.update(0)
    ''')
    assert lua.eval('table.concat(setters,",")') == 'enable,steer:0.9,limit:15'
    lua.execute('''
      now=1.1; car.position.x=1; script.update(0); now=4.1; script.update(0)
      now=4.2; script.update(0); now=4.3; script.update(0)
    ''')
    assert lua.eval('files[terminal].ai_steer_multiplier') == .9
    assert lua.eval('setters[#setters]') == 'steer:1'
    assert lua.eval('files[terminal].motion_probe_passed') is True


@pytest.mark.parametrize('value', [.5, 1.1, -1])
def test_arbitrary_ai_steer_multiplier_is_rejected(value):
    lua, _ = motion_service()
    lua.globals().request.ai_steer_multiplier = value
    lua.execute('script.update(0)')
    assert lua.eval('#setters') == 0


def test_collect30_keeps_owned_camera_and_separate_distance_time_bounds():
    lua, _ = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-collect30'; request.hold_seconds=30
      request.ai_steer_multiplier=.9
      physics.setAISteerMultiplier=function() end
      script.update(0)
      now=1.1; camera.last_capture_clock=now; car.position.x=1; script.update(0)
      now=31.1; camera.last_capture_clock=now; car.position.x=120; script.update(0)
      now=31.2; camera.last_capture_clock=now; script.update(0)
      now=31.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[terminal].command') == 'expert-collect30'
    assert lua.eval('files[terminal].maximum_position_delta_m') == 120


@pytest.mark.parametrize('mutation', [
    'setCamera(nil)', 'request.ai_steer_multiplier=1', 'camera.deadline=34',
    'request.hold_seconds=120',
])
def test_collect30_refuses_unowned_unbounded_or_short_camera_lease(mutation):
    lua, _ = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-collect30'; request.hold_seconds=30
      request.ai_steer_multiplier=.9
    ''')
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0 and lua.eval('calls') == 0


def fast_motion_service():
    lua, source = motion_service()
    lua.execute('''
      request.command='expert-motion-speed30'; request.speed_limit_kmh=30
      request.ai_steer_multiplier=.9
      physics.setAISteerMultiplier=function() end
    ''')
    return lua, source


@pytest.mark.parametrize('collection', [False, True])
def test_separate_speed60_profile_has_its_own_envelope_without_changing_speed30(collection):
    lua, _ = fast_motion_service()
    set_owned_camera(lua)
    lua.globals().collect60 = collection
    lua.execute('''
      request.command=collect60 and 'expert-collect30-speed60' or 'expert-motion-speed60'
      request.speed_limit_kmh=60; request.hold_seconds=collect60 and 30 or 3
      script.update(0)
      now=1.1; camera.last_capture_clock=now; car.position.x=1
      car.speedKmh=65; car.gas=.2; car.brake=0; script.update(0)
      now=collect60 and 31.1 or 4.1; camera.last_capture_clock=now
      car.position.x=collect60 and 500 or 60; script.update(0)
      now=now+.1; car.speedKmh=0; car.gas=0; car.brake=1; script.update(0)
      now=now+.1; script.update(0)
    ''')
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[terminal].requested_ai_speed_hint_kmh') == 60
    assert lua.eval('files[terminal].motion_hard_speed_kmh') == 80
    assert lua.eval('files[terminal].maximum_displacement_guard_m') == (720 if collection else 80)


def test_speed60_command_rejects_a_speed30_request_before_any_setter():
    lua, _ = fast_motion_service()
    lua.execute("request.command='expert-motion-speed60'; script.update(0)")
    assert lua.eval('#setters') == 0 and lua.eval('calls') == 0


def test_speed30_probe_keeps_separate_guard_and_verified_brake_handoff():
    lua, _ = fast_motion_service()
    lua.execute('''
      script.update(0)
      now=1.1; car.position.x=1; car.speedKmh=33; car.gas=.2; car.brake=0
      script.update(0)
      now=4.1; car.position.x=25; script.update(0)
      now=4.2; car.speedKmh=0; car.gas=0; car.brake=1; script.update(0)
      now=4.3; script.update(0)
    ''')
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[terminal].requested_ai_speed_hint_kmh') == 30
    assert lua.eval('files[terminal].motion_hard_speed_kmh') == 40
    assert lua.eval('files[terminal].maximum_displacement_guard_m') == 40


@pytest.mark.parametrize('residual_speed', [.01, .04587983712554, .099])
def test_motion_pause_leaves_margin_for_native_packet_before_freezing(residual_speed):
    lua, _ = fast_motion_service()
    lua.execute('''
      script.update(0)
      now=1.1; car.position.x=1; car.speedKmh=5; script.update(0)
      now=4.1; car.speedKmh=0.05; script.update(0)
    ''')
    lua.globals().car.speedKmh = residual_speed
    lua.execute('now=4.2; script.update(0)')
    assert lua.eval('sim.isPaused') is False
    assert lua.eval('car.isAIControlled') is False
    lua.execute('now=4.3; car.speedKmh=.009; script.update(0)')
    assert lua.eval('sim.isPaused') is True
    lua.execute('now=4.4; script.update(0)')
    assert lua.eval('files[terminal].status') == 'api_returned_true'


def test_speed30_overspeed_aborts_without_raising_original15_profile():
    lua, _ = fast_motion_service()
    lua.execute('script.update(0); now=1.1; car.speedKmh=41; script.update(0)')
    assert lua.eval('car.isAIControlled') is False
    lua.execute('now=1.2; car.speedKmh=0; script.update(0); now=1.3; script.update(0)')
    assert lua.eval('files[terminal].status') == 'api_returned_false'
    old, _ = motion_service()
    old.execute('request.speed_limit_kmh=30; script.update(0)')
    assert old.eval('#setters') == 0


def test_fast_collection_uses_owned_camera_and_rejects_arbitrary_speed():
    lua, _ = fast_motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-collect30-speed30'; request.hold_seconds=30
      script.update(0); now=1.1; camera.last_capture_clock=now
      car.position.x=1; car.speedKmh=33; script.update(0)
      now=31.1; camera.last_capture_clock=now; car.position.x=280; script.update(0)
      now=31.2; car.speedKmh=0; camera.last_capture_clock=now; script.update(0)
      now=31.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[terminal].maximum_displacement_guard_m') == 360
    bad, _ = fast_motion_service()
    bad.execute('request.speed_limit_kmh=60; script.update(0)')
    assert bad.eval('#setters') == 0


def recovery_service():
    lua, source = fast_motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-speed-recovery15'; request.hold_seconds=30
      request.warmup_seconds=3; request.recovery_target_kmh=15
    ''')
    return lua, source


def lateral_recovery_service():
    lua, source = motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-lateral-recovery15'; request.hold_seconds=30
      request.ai_steer_multiplier=.9; request.lateral_recovery=true
      car.wheelsOutside=1; physics.setAISteerMultiplier=function() end
    ''')
    return lua, source


@pytest.mark.parametrize('mutation', [
    'request.lateral_recovery=nil', 'car.wheelsOutside=2', 'car.wheelsOutside=0',
    'setCamera(nil)', 'request.hold_seconds=120', 'request.speed_limit_kmh=30',
    'request.ai_steer_multiplier=1'])
def test_lateral_recovery_is_a_separate_exact_owned_one_tyre_preset(mutation):
    lua, _ = lateral_recovery_service()
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0


@pytest.mark.parametrize('drive_tyres,stop_tyres,passed', [(0, 0, True),
    (1, 0, False), (0, 1, False)])
def test_lateral_recovery_requires_ai_movement_back_inside_and_inside_final_stop(
        drive_tyres, stop_tyres, passed):
    lua, _ = lateral_recovery_service()
    lua.execute('script.update(0)')
    assert lua.eval('car.isAIControlled') is True
    lua.execute(f'''
      now=1.1; camera.last_capture_clock=now; car.position.x=6; car.speedKmh=10
      car.wheelsOutside={drive_tyres}; script.update(0)
      now=31.1; camera.last_capture_clock=now; car.position.x=100; script.update(0)
      now=31.2; camera.last_capture_clock=now; car.speedKmh=0
      car.wheelsOutside={stop_tyres}; script.update(0)
      now=31.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].lateral_recovery15_passed') is passed
    assert lua.eval('files[terminal].motion_probe_passed') is passed
    assert lua.eval('car.isAIControlled') is False and lua.eval('sim.isPaused') is True
    assert lua.eval('files[terminal].motion_hard_speed_kmh') == 20


def test_recovery_retargets_once_and_requires_ai_overspeed_to_settle_before_handoff():
    lua, _ = recovery_service()
    lua.execute('''
      script.update(0)
      now=4.1; camera.last_capture_clock=now; car.position.x=20; car.speedKmh=30
      script.update(0)
    ''')
    assert lua.eval('setters[#setters]') == 'limit:15'
    assert lua.eval('setters[1]') == 'enable'
    lua.execute('''
      now=4.2; camera.last_capture_clock=now; car.speedKmh=27; script.update(0)
      now=5.2; camera.last_capture_clock=now; car.speedKmh=16; script.update(0)
      now=31.1; camera.last_capture_clock=now; car.position.x=140; script.update(0)
      now=31.2; camera.last_capture_clock=now; car.speedKmh=0; script.update(0)
      now=31.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].status') == 'api_returned_true'
    assert lua.eval('files[terminal].recovery15_passed') is True
    assert lua.eval('files[terminal].label_speed_hint_kmh') == 15
    assert lua.eval('files[terminal].telemetry[1].teacher_speed_hint_kmh') == 30
    assert lua.eval('files[terminal].telemetry[2].teacher_speed_hint_kmh') == 15
    assert lua.eval("table.concat(setters,',')").count('limit:15') == 1


def test_retarget_is_not_issued_after_session_identity_change():
    lua, _ = recovery_service()
    lua.execute('''
      script.update(0); now=4.1; camera.last_capture_clock=now
      ac.getCarID=function() return 'other' end; script.update(0)
    ''')
    assert lua.eval('car.isAIControlled') is False
    assert 'limit:15' not in lua.eval("table.concat(setters,',')")


def test_unknown_retarget_is_not_repeated_and_produces_unknown_receipt():
    lua, _ = recovery_service()
    lua.execute('''
      script.update(0); retargetCalls=0
      physics.setAITopSpeed=function(index,limit)
        if limit==15 then retargetCalls=retargetCalls+1; error('uncertain retarget') end
      end
      now=4.1; camera.last_capture_clock=now; script.update(0)
      now=4.2; camera.last_capture_clock=now; script.update(0)
      now=4.3; camera.last_capture_clock=now; script.update(0)
      now=4.4; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('retargetCalls') == 1
    assert lua.eval('files[terminal].status') == 'api_outcome_unknown'


@pytest.mark.parametrize('mutation', ['request.recovery_target_kmh=0',
                                     'request.warmup_seconds=4'])
def test_recovery_is_a_fixed_task_not_an_arbitrary_stop_command(mutation):
    lua, _ = recovery_service()
    lua.execute(mutation+'; script.update(0)')
    assert lua.eval('#setters') == 0


def test_warmup_overspeed_alone_cannot_qualify_the_recovery_phase():
    lua, _ = recovery_service()
    lua.execute('''
      script.update(0)
      now=4.1; camera.last_capture_clock=now; car.speedKmh=30; car.position.x=20
      script.update(0)
      now=4.2; camera.last_capture_clock=now; car.speedKmh=16; script.update(0)
      now=31.1; camera.last_capture_clock=now; script.update(0)
      now=31.2; camera.last_capture_clock=now; car.speedKmh=0; script.update(0)
      now=31.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].recovery15_passed') is False


@pytest.mark.parametrize('post_target_speed,passed', [(19, True), (18, False)])
def test_speed20_recovery_uses_fixed25_guard_and_observed18_to17_transition(
        post_target_speed, passed):
    lua, _ = recovery_service()
    lua.execute('''
      request.command='expert-speed20-recovery15'; request.speed_limit_kmh=20
      script.update(0)
      now=4.1; camera.last_capture_clock=now; car.position.x=15; car.speedKmh=20.5
      script.update(0)
    ''')
    lua.execute(f'''
      now=4.2; camera.last_capture_clock=now; car.speedKmh={post_target_speed}; script.update(0)
      now=5.2; camera.last_capture_clock=now; car.speedKmh=16; script.update(0)
      now=31.1; camera.last_capture_clock=now; car.position.x=120; script.update(0)
      now=31.2; camera.last_capture_clock=now; car.speedKmh=0; script.update(0)
      now=31.3; camera.last_capture_clock=now; script.update(0)
    ''')
    assert lua.eval('files[terminal].recovery15_passed') is passed
    assert lua.eval('files[terminal].motion_hard_speed_kmh') == 25
    assert lua.eval('files[terminal].maximum_displacement_guard_m') == 220
    assert lua.eval('files[terminal].recovery_start_threshold_kmh') == 18
    assert lua.eval('files[terminal].requested_ai_speed_hint_kmh') == 20
    assert lua.eval('files[terminal].label_speed_hint_kmh') == 15
    assert lua.eval("table.concat(setters,',')").count('limit:15') == 1


def test_speed20_recovery_hard_guard_precedes_retarget_and_rejects_wrong_payload():
    lua, _ = recovery_service()
    lua.execute('''
      request.command='expert-speed20-recovery15'; request.speed_limit_kmh=20
      script.update(0); now=4.1; camera.last_capture_clock=now
      car.speedKmh=25.1; script.update(0)
    ''')
    assert lua.eval('car.isAIControlled') is False
    assert 'limit:15' not in lua.eval("table.concat(setters,',')")
    wrong, _ = recovery_service()
    wrong.execute('request.speed_limit_kmh=20; script.update(0)')
    assert wrong.eval('#setters') == 0


@pytest.mark.parametrize('origin', ['nil', '{x=0,y=0,z=1024}', '{x=0,y=0,z=0/0}'])
def test_origin_shift_is_read_only_diagnostic_and_missing_is_not_zero(origin):
    lua, _ = service()
    lua.execute("files['mock/flybrain/csp-session/request.json']=nil")
    lua.execute('sim.originShift='+origin+'; script.update(0)')
    status = lua.eval("files['mock/flybrain/csp-session/status.json']")
    if origin == '{x=0,y=0,z=1024}':
        assert list(status['renderer_origin_shift_diagnostic'].values()) == [0, 0, 1024]
    else:
        assert status['renderer_origin_shift_diagnostic'] is None
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('scene,root', [('full_scene', 'sceneRoot:yes'),
                                     ('track_only', 'trackRoot:yes')])
def test_paused_camera_motion_probe_selects_only_requested_scene_without_actuation(scene, root):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      ac.findNodes=function(name) selectedScene=name; error('RENDER_ENTRY_SENTINEL') end
      probe={id='rotation',frames=51,period_seconds=.04,deadline_seconds=8,exposure=.35,
        image_format='bmp',shared_transport=true,camera_motion_probe={yaw_left_rad_s=.2,
          static_pre_seconds=.6,motion_seconds=1.2}}
    ''')
    lua.globals().probe['camera_motion_probe']['scene'] = scene
    lua.execute("files['mock/flybrain/csp-camera/request.json']=probe; script.update(0)")
    assert lua.eval('selectedScene') == root
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('mutation', ['sim.isPaused=false', 'car.isAIControlled=true',
    'car.speedKmh=1', 'probe.frames=81', 'probe.camera_motion_probe.yaw_left_rad_s=.26',
    "probe.camera_motion_probe.scene='arbitrary'", 'probe.camera_motion_probe.motion_seconds=4'])
def test_camera_motion_probe_rejects_unbounded_or_live_conditions_before_renderer(mutation):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      ac.findNodes=function(name) selectedScene=name; error('RENDER_ENTRY_SENTINEL') end
      probe={id='rotation',frames=51,period_seconds=.04,deadline_seconds=8,exposure=.35,
        image_format='bmp',shared_transport=true,camera_motion_probe={yaw_left_rad_s=.2,
          static_pre_seconds=.6,motion_seconds=1.2,scene='full_scene'}}
    ''')
    lua.execute(mutation+"; files['mock/flybrain/csp-camera/request.json']=probe; script.update(0)")
    assert lua.eval('selectedScene') is None
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('mutation', ["probe.camera_motion_probe.front_mount='unknown'",
    "probe.camera_motion_probe.scene='track_only'", 'probe.frames=9',
    'probe.camera_motion_probe.yaw_left_rad_s=.1',
    "ac.getCarID=function() return 'other_car' end"])
def test_exterior_eye_probe_rejects_wrong_identity_scene_motion_or_budget(mutation):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      ac.findNodes=function(name) selectedScene=name; error('RENDER_ENTRY_SENTINEL') end
      probe={id='exterior',frames=1,period_seconds=.1,deadline_seconds=4,exposure=.35,
        image_format='bmp',shared_transport=true,camera_motion_probe={yaw_left_rad_s=0,
          static_pre_seconds=.6,motion_seconds=1.2,scene='full_scene',
          front_mount='bmw_z4_gt3_bumper_v1'}}
    ''')
    lua.execute(mutation+"; files['mock/flybrain/csp-camera/request.json']=probe; script.update(0)")
    assert lua.eval('selectedScene') is None
    assert lua.eval('calls') == 0


def test_valid_exterior_eye_enters_only_paused_full_scene_renderer():
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false
      ac.findNodes=function(name) selectedScene=name; error('RENDER_ENTRY_SENTINEL') end
      probe={id='exterior',frames=1,period_seconds=.1,deadline_seconds=4,exposure=.35,
        image_format='bmp',shared_transport=true,camera_motion_probe={yaw_left_rad_s=0,
          static_pre_seconds=.6,motion_seconds=1.2,scene='full_scene',
          front_mount='bmw_z4_gt3_bumper_v1'}}
      files['mock/flybrain/csp-camera/request.json']=probe; script.update(0)
    ''')
    assert lua.eval('selectedScene') == 'sceneRoot:yes'
    assert lua.eval('calls') == 0


@pytest.mark.parametrize('mutation', ["request.camera_sensor_layout='unknown'",
    'request.mmf_metadata=false', 'request.host_persistence_protocol=nil',
    'request.frames=1201', 'sim.isPaused=false', 'car.isAIControlled=true',
    'car.speedKmh=1', 'car.gas=.03', 'car.brake=.9',
    "ac.getCarID=function() return 'other_car' end",
    'request.camera_motion_probe={}', 'request.allow_ideal_line_probe=true'])
def test_normal_exterior_stream_rejects_unregistered_or_unowned_start(mutation):
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false; mmfCalls=0
      ac.readMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('MMF_SENTINEL') end
      ac.writeMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('MMF_SENTINEL') end
      request={id='mmf-stream-'..string.rep('a',32), frames=8,
        camera_sensor_layout='exterior_front_v3',mmf_metadata=true,
        host_persistence_protocol='immutable_ram_v3',shared_transport=true,
        image_format='bmp',period_seconds=.08,deadline_seconds=10,exposure=.35}
    ''')
    lua.execute(mutation+"; files['mock/flybrain/csp-camera/request.json']=request;"
                " script.update(0)")
    assert lua.eval('mmfCalls') == 0 and lua.eval('calls') == 0


def test_normal_exterior_stream_valid_owned_start_reaches_mmf_without_actuation():
    lua, _ = service()
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      sim.isPaused=true; car.isAIControlled=false; mmfCalls=0
      ac.readMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('MMF_SENTINEL') end
      ac.writeMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('MMF_SENTINEL') end
      request={id='mmf-stream-'..string.rep('a',32), frames=8,
        camera_sensor_layout='exterior_front_v3',mmf_metadata=true,
        host_persistence_protocol='immutable_ram_v3',shared_transport=true,
        image_format='bmp',period_seconds=.08,deadline_seconds=10,exposure=.35}
      files['mock/flybrain/csp-camera/request.json']=request; script.update(0)
    ''')
    assert lua.eval('mmfCalls') == 1 and lua.eval('calls') == 0
