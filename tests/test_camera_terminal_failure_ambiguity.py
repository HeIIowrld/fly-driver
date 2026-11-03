"""Reproduce two distinct causes of the same camera terminal/status mismatch."""
from test_ac_lua_session_receipts import service, handled_luajit_errors_are_not_python_fatal_signals  # noqa: F401
from pathlib import Path


def camera_service(tmp_path, source='integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
    original = Path(source).read_text(encoding='utf8')
    # This hook exists only in a temporary Lua mock; deployed Lua is unchanged.
    hook = '''\nfunction script.inject_test_capture()
      lastID='req'; active={directory='mock/flybrain/csp-camera/req',
        request={id='req'}, frames=16, views={}}
    end\n'''
    candidate = tmp_path/'camera_with_test_hook.lua'
    candidate.write_text(original+hook, encoding='utf8')
    lua, _ = service(candidate)
    lua.execute("files['mock/flybrain/csp-session/request.json']=nil; script.inject_test_capture(); files['mock/flybrain/csp-camera/req/stop']=''")
    return lua


def test_failed_stopped_status_write_can_follow_successful_terminal(tmp_path):
    lua = camera_service(tmp_path)
    lua.execute("failures['mock/flybrain/csp-camera/status.json']=1; script.update(0)")
    assert lua.eval("files['mock/flybrain/csp-camera/req/result.json'].status") == 'stopped'
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].state") == 'failed'
    assert 'Cannot commit camera metadata' in lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")
    assert lua.eval('calls') == 0


def test_later_failed_clock_reply_can_produce_same_mismatch(tmp_path):
    lua = camera_service(tmp_path)
    lua.execute("script.update(0); now=1.5; files['mock/flybrain/csp-camera/clock-ping.json']={nonce='n'}; failures['mock/flybrain/csp-camera/clock-reply.json']=1; script.update(0)")
    assert lua.eval("files['mock/flybrain/csp-camera/req/result.json'].status") == 'stopped'
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].state") == 'failed'
    assert 'Cannot commit camera metadata' in lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")
    assert lua.eval('calls') == 0


def test_diagnostic_candidate_distinguishes_the_two_paths(tmp_path):
    from scripts.build_camera_write_diagnostic_candidate import build, OLD, NEW, SOURCE
    candidate = Path('integrations/ac_csp/candidates/FlyBrainCamera-write-path-diagnostic.lua')
    assert candidate.read_bytes() == build()
    assert candidate.read_text(encoding='utf8').replace(NEW, OLD) == SOURCE.read_text(encoding='utf8')
    lua = camera_service(tmp_path, str(candidate))
    lua.execute("failures['mock/flybrain/csp-camera/status.json']=1; script.update(0)")
    assert 'csp-camera/status.json' in lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")
    lua = camera_service(tmp_path, str(candidate))
    lua.execute("script.update(0); now=1.5; files['mock/flybrain/csp-camera/clock-ping.json']={nonce='n'}; failures['mock/flybrain/csp-camera/clock-reply.json']=1; script.update(0)")
    assert 'csp-camera/clock-reply.json' in lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")
    assert lua.eval('calls') == 0
