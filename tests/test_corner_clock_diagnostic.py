from pathlib import Path

import pytest

from tests.test_ac_lua_session_receipts import (
    handled_luajit_errors_are_not_python_fatal_signals,  # noqa: F401
    service,
)


@pytest.mark.parametrize("frames,accepted", [(550, True), (601, False)])
def test_extended_diagnostic_frame_limit_keeps_offline_guard(frames, accepted):
    lua, _ = service()
    path = Path("integrations/ac_csp/candidates/FlyBrainCamera-corner-clock-diagnostic-v1.lua")
    lua.execute(path.read_text(encoding="utf8"))
    lua.globals().test_frames = frames
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='probe', frames=test_frames,
        period_seconds=.08, deadline_seconds=50, exposure=.35, image_format='png',
        airflow_diagnostic=true, deferred_save_probe=true}
      sim.isPaused=true; sim.isOnlineRace=false; car.isAIControlled=false
      car.speedKmh=0; car.gas=0; car.brake=1
      ac.findNodes=function() error('reached_geometry_setup') end
      script.update(0)
    """)
    reason = lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")
    assert ("reached_geometry_setup" in reason) is accepted
    if not accepted:
        assert "bounded offline brake-held" in reason
    assert lua.eval("calls") == 0
