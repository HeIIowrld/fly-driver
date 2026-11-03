from pathlib import Path

from tests.test_ac_lua_session_receipts import (
    handled_luajit_errors_are_not_python_fatal_signals,  # noqa: F401
    service,
)


def test_deferred_camera_rejects_moving_or_unbounded_session():
    lua, _ = service()
    candidate = (
        Path("integrations/ac_csp/candidates") / "FlyBrainCamera-deferred-save-diagnostic-v1.lua"
    )
    lua.execute(candidate.read_text(encoding="utf8"))
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='probe', frames=4,
        period_seconds=.1, deadline_seconds=5, exposure=.35, image_format='png',
        deferred_save_probe=true, verify_png_reference=true}
      sim.isPaused=false; car.isAIControlled=false
      script.update(0)
    """)
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].state") == "failed"
    assert "bounded paused offline" in lua.eval(
        "files['mock/flybrain/csp-camera/status.json'].reason"
    )
    assert lua.eval("calls") == 0


def test_deferred_paired_png_passes_request_guards_in_paused_session():
    lua, _ = service()
    candidate = (
        Path("integrations/ac_csp/candidates") / "FlyBrainCamera-deferred-save-diagnostic-v1.lua"
    )
    lua.execute(candidate.read_text(encoding="utf8"))
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='probe', frames=4,
        period_seconds=.1, deadline_seconds=5, exposure=.35, image_format='png',
        deferred_save_probe=true, verify_png_reference=true}
      sim.isPaused=true; sim.isOnlineRace=false; car.isAIControlled=false
      ac.findNodes=function() error('reached_geometry_setup') end
      script.update(0)
    """)
    assert "reached_geometry_setup" in lua.eval(
        "files['mock/flybrain/csp-camera/status.json'].reason"
    )
    assert lua.eval("calls") == 0


def test_deferred_motion_rejects_unpaused_vehicle():
    lua, _ = service()
    path = (
        Path("integrations/ac_csp/candidates") / "FlyBrainCamera-deferred-motion-diagnostic-v1.lua"
    )
    lua.execute(path.read_text(encoding="utf8"))
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='probe', frames=26,
        period_seconds=.08, deadline_seconds=10, exposure=.35, image_format='png',
        deferred_save_probe=true, verify_png_reference=false,
        camera_motion_probe={yaw_left_rad_s=.2, static_pre_seconds=.6,
          motion_seconds=1.2, scene='full_scene'}}
      sim.isPaused=false; car.isAIControlled=false
      script.update(0)
    """)
    assert "bounded paused offline" in lua.eval(
        "files['mock/flybrain/csp-camera/status.json'].reason"
    )
    assert lua.eval("calls") == 0


def test_owned_ai_deferred_probe_requires_diagnostic_capability():
    lua, _ = service()
    folder = Path("integrations/ac_csp/candidates")
    candidate = folder / "FlyBrainCamera-deferred-owned-ai-diagnostic-v1.lua"
    lua.execute(candidate.read_text(encoding="utf8"))
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='probe', frames=4,
        period_seconds=.1, deadline_seconds=5, exposure=.35, image_format='png',
        deferred_save_probe=true}
      sim.isPaused=true; car.isAIControlled=false
      script.update(0)
    """)
    assert "requires owned airflow diagnostic" in lua.eval(
        "files['mock/flybrain/csp-camera/status.json'].reason"
    )
    assert lua.eval("calls") == 0


def test_frame_clock_candidate_rejects_unowned_deferred_probe():
    lua, _ = service()
    folder = Path("integrations/ac_csp/candidates")
    candidate = folder / "FlyBrainCamera-frame-clock-diagnostic-v1.lua"
    lua.execute(candidate.read_text(encoding="utf8"))
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='probe', frames=4,
        period_seconds=.1, deadline_seconds=5, exposure=.35, image_format='png',
        deferred_save_probe=true}
      sim.isPaused=true; car.isAIControlled=false
      script.update(0)
    """)
    assert "requires owned airflow diagnostic" in lua.eval(
        "files['mock/flybrain/csp-camera/status.json'].reason"
    )
    assert lua.eval("calls") == 0
