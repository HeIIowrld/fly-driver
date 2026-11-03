from pathlib import Path
import pytest


def _setup():
    lupa = pytest.importorskip('lupa.luajit21')
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    module = lua.execute(Path('integrations/ac_csp/candidates/TrackPhysicsMarkerDiagnostic.lua').read_text(encoding='utf8'))
    fixture = lua.eval("""function()
      local callbacks={} local disposed={} local ticks=0
      local ac={getSim=function() return {time=1200} end,
        onTrackPointCrossed=function(car,progress,cb)
          assert(car==0); callbacks[progress]=cb
          return function() disposed[progress]=true end
        end}
      local function clock() ticks=ticks+1; return 10+ticks*.001 end
      return ac,clock,callbacks,disposed
    end""")
    return lua, module, fixture()


def test_marker_observations_are_bounded_and_disposed():
    lua,m,(ac,clock,callbacks,disposed)=_setup()
    observer=m.start(ac,clock,'gen',lua.table_from([.2,.3]))
    callbacks[.2](0,1198)
    callbacks[.2](0,1199)
    event=observer.snapshot(observer)['events'][1]
    assert event['physics_event_ms']==1198 and event['callback_sim_time_ms']==1200
    assert not event['actor_admissible'] and not event['physical_sync_verified']
    observer.stop(observer)
    assert disposed[.2] and disposed[.3]
    callbacks[.3](0,1201)
    assert len(observer.snapshot(observer)['events'])==1


def test_marker_input_and_partial_registration_failure():
    lua,m,(ac,clock,callbacks,disposed)=_setup()
    with pytest.raises(Exception, match='unique markers'):
        m.start(ac,clock,'gen',lua.table_from([.2,.2]))
    with pytest.raises(Exception, match='normalized marker'):
        m.start(ac,clock,'gen',lua.table_from([1.1]))
    broken=lua.eval("""function(ac) local old=ac.onTrackPointCrossed; local n=0;
      ac.onTrackPointCrossed=function(...) n=n+1; if n==2 then error('second registration') end; return old(...) end; return ac end""")(ac)
    with pytest.raises(Exception, match='second registration'):
        m.start(broken,clock,'gen',lua.table_from([.2,.3]))
    assert disposed[.2]


def test_integrated_candidate_compiles_and_is_diagnostic_only():
    lua,_,_= _setup()
    text=Path('integrations/ac_csp/candidates/FlyBrainCamera-physics-marker-diagnostic-v1.lua').read_text(encoding='utf8')
    ok,err=lua.eval('function(s) local f,e=loadstring(s); return f~=nil,e end')(text)
    assert ok,err
    assert 'Physics markers require owned motion diagnostic' in text
    assert 'active.track_marker_observer:stop()' in text
    assert 'metadata.physics_marker_diagnostic' in text
