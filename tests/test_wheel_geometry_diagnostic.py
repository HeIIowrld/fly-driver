from pathlib import Path

import pytest


def test_wheel_geometry_capture_retains_indices_and_rejects_bad_values():
    lupa = pytest.importorskip('lupa')
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    module = lua.execute(Path('integrations/ac_csp/candidates/WheelGeometryDiagnostic.lua').read_text(encoding='utf8'))
    fixture = lua.eval("""function()
      local t=0
      local function clock() t=t+.001; return t end
      local function vec(x,y,z) return {x=x,y=y,z=z} end
      local wheels={}
      for i=0,3 do wheels[i]={load=100+i,position=vec(i,0,1),contactPoint=vec(i,0,0),outside=vec(0,1,0)} end
      local car={position=vec(0,0,0),look=vec(1,0,0),side=vec(0,1,0),up=vec(0,0,1),wheels=wheels,timestamp=100}
      local ac={getCar=function(i) assert(i==0); return car end,getSim=function() return {frame=12} end}
      return ac,clock,car
    end""")
    ac, clock, car = fixture()
    record = module.capture(ac, clock, 'generation')
    assert record['actor_admissible'] is False and record['source_verified'] is False
    assert [record['wheels'][i]['native_index'] for i in range(1, 5)] == [0, 1, 2, 3]
    assert [record['wheels'][i]['load_candidate'] for i in range(1, 5)] == [100, 101, 102, 103]
    assert record['finished_source_clock'] > record['started_source_clock']
    car['wheels'][2]['load'] = -1
    with pytest.raises(Exception, match='wheel load'):
        module.capture(ac, clock, 'generation')


def test_candidate_compiles_and_has_both_stationary_gates():
    luajit = pytest.importorskip('lupa.luajit21')
    text = Path('integrations/ac_csp/candidates/FlyBrainCamera-wheel-geometry-diagnostic-v1.lua').read_text(encoding='utf8')
    compile_lua = luajit.LuaRuntime(unpack_returned_tuples=True).eval('function(s) local f,e=loadstring(s); return f ~= nil,e end')
    success, error = compile_lua(text)
    assert success, error
    assert 'Wheel geometry diagnostic requires paused offline brake-held' in text
    assert 'Wheel geometry diagnostic lost paused offline brake-held' in text
    assert 'metadata.aid_probe_only = true' in text
    assert 'metadata.wheel_geometry_diagnostic = wheelGeometryDiagnostic.capture' in text


def _geometry_fixture():
    return {
        "schema": "flybrain.csp.wheel-geometry.diagnostic.v1",
        "actor_admissible": False, "source_verified": False,
        "body_forward_candidate": [1, 0, 0], "body_up_candidate": [0, 0, 1],
        "wheels": [
            {"native_index": i, "position_world": [x, y, 0],
             "contact_world": [x, y, -0.3], "outside_world": [0, sign, 0],
             "load_candidate": 1000 + i}
            for i, (x, y, sign) in enumerate(((1, 1, 1), (1, -1, -1), (-1, 1, 1), (-1, -1, -1)))
        ],
    }


def test_offline_geometry_identifies_permutation_without_qualifying_actor():
    from flybrain_drive.wheel_geometry_diagnostic import analyze_wheel_geometry
    d = _geometry_fixture()
    r = analyze_wheel_geometry(d)
    assert r['observed_labels'] == ['FL', 'FR', 'RL', 'RR']
    assert r['matches_fl_fr_rl_rr'] and not r['native_order_registered']
    d['wheels'][0], d['wheels'][1] = d['wheels'][1], d['wheels'][0]
    for i, w in enumerate(d['wheels']):
        w['native_index'] = i
    r = analyze_wheel_geometry(d)
    assert r['observed_labels'] == ['FR', 'FL', 'RL', 'RR']
    assert not r['matches_fl_fr_rl_rr']


def test_offline_geometry_rejects_invalid_basis_and_ambiguous_positions():
    from flybrain_drive.wheel_geometry_diagnostic import analyze_wheel_geometry
    d = _geometry_fixture()
    d['body_up_candidate'] = [1, 0, 0]
    with pytest.raises(ValueError, match='orthogonal'):
        analyze_wheel_geometry(d)
    d = _geometry_fixture()
    d['wheels'][0]['position_world'] = d['wheels'][1]['position_world']
    with pytest.raises(ValueError):
        analyze_wheel_geometry(d)


def test_offline_geometry_rejects_wrong_outside_vector_and_actor_claim():
    from flybrain_drive.wheel_geometry_diagnostic import analyze_wheel_geometry
    d = _geometry_fixture()
    d['wheels'][0]['outside_world'] = [0, -1, 0]
    with pytest.raises(ValueError, match='outside'):
        analyze_wheel_geometry(d)
    d = _geometry_fixture()
    d['actor_admissible'] = True
    with pytest.raises(ValueError, match='diagnostic'):
        analyze_wheel_geometry(d)


def test_motion_candidate_compiles_and_requires_owned_probe():
    luajit = pytest.importorskip('lupa.luajit21')
    text = Path('integrations/ac_csp/candidates/FlyBrainCamera-wheel-motion-diagnostic-v1.lua').read_text(encoding='utf8')
    compile_lua = luajit.LuaRuntime(unpack_returned_tuples=True).eval('function(s) local f,e=loadstring(s); return f ~= nil,e end')
    success, error = compile_lua(text)
    assert success, error
    assert 'expertProbe.camera_request_id == active.request.id' in text
    assert 'Airflow diagnostic lost paused state or its owned motion probe' in text
    assert 'metadata.wheel_geometry_diagnostic = wheelGeometryDiagnostic.capture' in text
    assert 'metadata.aid_probe_only = true' in text


def test_clock_candidate_records_both_sim_and_physics_times():
    lupa = pytest.importorskip('lupa')
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    module = lua.execute(Path('integrations/ac_csp/candidates/WheelGeometryClockDiagnostic.lua').read_text(encoding='utf8'))
    fixture = lua.eval("""function()
      local t=0
      local function clock() t=t+.001; return t end
      local function v(x,y,z) return {x=x,y=y,z=z} end
      local w={} for i=0,3 do w[i]={load=100+i,position=v(i,0,1),contactPoint=v(i,0,0),outside=v(0,1,0)} end
      local car={position=v(0,0,0),look=v(1,0,0),side=v(0,1,0),up=v(0,0,1),wheels=w,timestamp=100}
      local sim={frame=12,time=104}; return {getCar=function() return car end,getSim=function() return sim end},clock
    end""")
    ac, clock = fixture()
    r = module.capture(ac, clock, 'g')
    assert r['car_physics_timestamp_ms'] == 100 and r['car_physics_timestamp_after_ms'] == 100
    assert r['sim_time_before_ms'] == 104 and r['sim_time_after_ms'] == 104
    assert r['actor_admissible'] is False and r['source_verified'] is False
    assert r['state_update_domain'] == 'csp_graphics_frame'
    assert r['sim_update_domain'] == 'csp_graphics_frame'
    assert r['native_tick_link_verified'] is False


def test_clock_motion_candidate_compiles_with_owned_gate():
    luajit = pytest.importorskip('lupa.luajit21')
    text = Path('integrations/ac_csp/candidates/FlyBrainCamera-wheel-clock-diagnostic-v1.lua').read_text(encoding='utf8')
    check = luajit.LuaRuntime(unpack_returned_tuples=True).eval('function(s) local f,e=loadstring(s); return f ~= nil,e end')
    ok, error = check(text)
    assert ok, error
    assert 'expertProbe.camera_request_id == active.request.id' in text


def test_clock_candidate_v2_marks_graphics_domain_and_compiles():
    luajit = pytest.importorskip('lupa.luajit21')
    text = Path('integrations/ac_csp/candidates/FlyBrainCamera-wheel-clock-diagnostic-v2.lua').read_text(encoding='utf8')
    check = luajit.LuaRuntime(unpack_returned_tuples=True).eval('function(s) local f,e=loadstring(s); return f ~= nil,e end')
    ok, error = check(text)
    assert ok, error
    assert "state_update_domain='csp_graphics_frame'" in text
    assert 'native_tick_link_verified=false' in text
