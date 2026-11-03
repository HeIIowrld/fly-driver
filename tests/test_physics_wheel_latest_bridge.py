from pathlib import Path
import pytest


def _module():
    luajit = pytest.importorskip('lupa.luajit21')
    lua = luajit.LuaRuntime(unpack_returned_tuples=True)
    return lua, lua.execute(Path('integrations/ac_csp/candidates/PhysicsWheelLatestBridge.lua').read_text(encoding='utf8'))


def test_commit_and_generation_preserve_original_wheel_order():
    lua, m = _module()
    slot = lua.table_from({'commit': 0})
    sample = lua.table_from({'sequence': 1, 'elapsed': .003, 'wheel_load_native_order': lua.table_from([100, 200, 300, 400])})
    m.publish(slot, 72, sample)
    row, err = m.read(slot, 72)
    assert err is None and row['sequence'] == 1 and row['elapsed'] == pytest.approx(.003)
    assert [row['wheel_load_native_order'][i] for i in range(1, 5)] == [100, 200, 300, 400]
    assert not row['atomicity_verified'] and not row['physical_time_verified'] and not row['actor_admissible']
    row, err = m.read(slot, 73)
    assert row is None and err == 'generation_mismatch'
    slot['commit'] = 3
    row, err = m.read(slot, 72)
    assert row is None and err == 'uncommitted'


def test_changed_sequence_and_invalid_load_fail_closed():
    lua, m = _module()
    tearing = lua.eval("""function()
      local reads=0
      return setmetatable({generation=7,sample_sequence=1,relative_elapsed=.003,wheel0=1,wheel1=1,wheel2=1,wheel3=1},
        {__index=function(_,k) if k=='commit' then reads=reads+1; return reads*2 end end})
    end""")()
    row, err = m.read(tearing, 7)
    assert row is None and err == 'changed'
    slot = lua.table_from({'commit': 0})
    sample = lua.table_from({'sequence': 1, 'elapsed': .003, 'wheel_load_native_order': lua.table_from([1, 2, -1, 4])})
    with pytest.raises(Exception, match='nonnegative loads'):
        m.publish(slot, 7, sample)
    assert slot['commit'] == 0


def test_channel_key_is_car_specific_and_namespace_explicit_to_caller():
    lua, m = _module()
    ac = lua.eval("""{StructItem={key=function(k) return k end,uint32=function() return 0 end,
        float=function() return 0 end,double=function() return 0 end}}""")
    assert m.layout(ac, 0)[1] != m.layout(ac, 1)[1]
    with pytest.raises(Exception, match='car index'):
        m.layout(ac, -1)


def test_csp_transport_candidate_compiles_and_is_explicitly_isolated():
    lua, _ = _module()
    text = Path('integrations/ac_csp/candidates/FlyBrainCamera-wheel-bridge-diagnostic-v1.lua').read_text(encoding='utf8')
    ok, error = lua.eval('function(s) local f,e=loadstring(s); return f ~= nil,e end')(text)
    assert ok, error
    assert 'id>=1000000000 and id<=2147483647' in text
    assert 'same_context_only=true, physics_thread_verified=false, actor_admissible=false' in text
    assert 'Wheel geometry diagnostic requires paused offline brake-held' in text
