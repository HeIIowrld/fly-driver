from pathlib import Path
import pytest


def _setup():
    lupa = pytest.importorskip('lupa.luajit21')
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    module = lua.execute(Path('integrations/ac_csp/candidates/PhysicsTickWheelProbe.lua').read_text(encoding='utf8'))
    fixture = lua.eval("""function()
      local wheels={} for i=0,3 do wheels[i]={load=100+i} end
      local car={speedKmh=15,wheels=wheels,gForces={x=1,y=2,z=3},localAngularVelocity={x=4,y=5,z=6}}
      return {accessCarPhysics=function() return car end},car
    end""")
    ac, car = fixture()
    return module, ac, car


def test_observer_retains_order_and_only_relative_time():
    module, ac, car = _setup()
    probe = module.new(2)
    probe.step(probe, ac, .003)
    probe.step(probe, ac, .004)
    first, second = probe['entries'][1], probe['entries'][2]
    assert [first['wheel_load_native_order'][i] for i in range(1, 5)] == [100, 101, 102, 103]
    assert first['sequence'] == 1 and second['sequence'] == 2
    assert second['elapsed'] == pytest.approx(.007)
    assert second['native_packet_id'] is None and not second['physical_time_verified']
    assert not second['actor_admissible'] and car['speedKmh'] == 15
    with pytest.raises(Exception, match='capacity exhausted'):
        probe.step(probe, ac, .003)
    with pytest.raises(Exception, match='explicit reset'):
        probe.step(probe, ac, .003)
    probe.reset(probe)
    assert probe['count'] == 0 and not probe['failed']


def test_bad_sample_latches_without_partial_record():
    module, ac, car = _setup()
    probe = module.new(4)
    car['wheels'][2]['load'] = -1
    with pytest.raises(Exception, match='wheel load'):
        probe.step(probe, ac, .003)
    assert probe['count'] == 0 and probe['sequence'] == 0
    car['wheels'][2]['load'] = 1
    with pytest.raises(Exception, match='explicit reset'):
        probe.step(probe, ac, .003)
    probe.reset(probe)
    probe.step(probe, ac, .003)
    assert probe['count'] == 1


def test_capacity_and_dt_are_explicitly_bounded():
    module, ac, _ = _setup()
    for capacity in (0, 2049, 1.5):
        with pytest.raises(Exception, match='capacity'):
            module.new(capacity)
    probe = module.new(2)
    with pytest.raises(Exception, match='physics dt'):
        probe.step(probe, ac, .5)
