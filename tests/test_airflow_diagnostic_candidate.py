from pathlib import Path

import pytest


def test_lua_diagnostic_candidate_reads_once_and_never_claims_actor_admission():
    lupa = pytest.importorskip('lupa')
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    script = Path('integrations/ac_csp/candidates/RelativeAirflowDiagnostic.lua').read_text(
        encoding='utf8')
    module = lua.execute(script)
    fixture = lua.eval("""function()
      local n = 0
      local function clock() n=n+1; return 20+n*.001 end
      local function v(x,y,z) return {x=x,y=y,z=z} end
      local car = {velocity=v(10,0,0),look=v(1,0,0),side=v(0,1,0),
                   up=v(0,0,1),timestamp=1234}
      local sim = {windVelocityKmh={x=3.6,y=7.2},windSpeedKmh=8,frame=8}
      local ac = {getCar=function(_) return car end, getSim=function() return sim end,
                  getWindVelocity=function() return v(1,0,2) end}
      return ac,clock
    end""")
    ac, clock = fixture()
    record = module.capture(ac, clock, 'gen')
    assert record['schema'] == 'flybrain.csp.airflow.diagnostic.v1'
    assert record['actor_admissible'] is False and record['source_verified'] is False
    assert record['wind_velocity_candidate_m_s'][1] == 1
    assert record['vehicle_velocity_candidate_m_s'][1] == 10
    assert record['finished_source_clock'] > record['started_source_clock']
    with pytest.raises(Exception, match='generation'):
        module.capture(ac, clock, '')


def test_integrated_candidate_is_luajit_compilable_and_diagnostic_only():
    luajit = pytest.importorskip('lupa.luajit21')
    source = Path('integrations/ac_csp/candidates/FlyBrainCamera-airflow-diagnostic-v1.lua')
    text = source.read_text(encoding='utf8')
    compile_lua = luajit.LuaRuntime(unpack_returned_tuples=True).eval(
        'function(s) local f,e=loadstring(s); return f ~= nil,e end')
    success, message = compile_lua(text)
    assert success, message
    assert 'metadata.aid_probe_only = true' in text
    assert 'Airflow diagnostic lost paused, offline, brake-held conditions' in text


def test_motion_candidate_has_explicit_finite_brake_capture_gate():
    luajit = pytest.importorskip('lupa.luajit21')
    text = Path('integrations/ac_csp/candidates/FlyBrainCamera-airflow-motion-diagnostic-v1.lua').read_text(encoding='utf8')
    success, message = luajit.LuaRuntime(unpack_returned_tuples=True).eval(
        'function(s) local f,e=loadstring(s); return f ~= nil,e end')(text)
    assert success, message
    assert 'request.brake_pulse_capture and 600 or 300' in text
    assert 'request.deadline_seconds <= 55' in text
    assert 'request.airflow_diagnostic == true' in text
