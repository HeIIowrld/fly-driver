"""Mock-only release observation; never asserts old engine termination."""
from test_ac_lua_session_receipts import service, handled_luajit_errors_are_not_python_fatal_signals  # noqa: F401
from scripts.build_camera_release_witness_candidate import DEST, build


def test_release_witness_describes_only_own_generation():
    assert DEST.read_bytes()==build()
    lua,_=service(DEST)
    lua.execute("files['mock/flybrain/csp-session/request.json']=nil; script.update(0); releaseCallback()")
    w=lua.eval("files['mock/flybrain/csp-camera/release-17_1.json']")
    assert w['generation']=='17:1' and w['active_before'] is False and w['active_after'] is False
    assert w['prior_generation_resolved'] is False and w['engine_exit_verified'] is False and w['actor_admissible'] is False
    assert lua.eval('calls')==0


def test_repeat_orphan_observation_never_overwrites_and_does_not_become_terminal():
    lua,_=service(DEST)
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='oldreq',expected_generation='oldgen'}
      files['mock/flybrain/csp-camera/oldreq/stop']=''
      files['mock/flybrain/csp-camera/oldreq/orphan-observation.json']={schema='flybrain.csp.camera.orphan_observation.v1',status='prior_generation_unresolved',request_id='oldreq',expected_generation='oldgen',observer_generation='priorobserver',actor_admissible=false}
      script.update(0); now=2; script.update(0)
    ''')
    assert lua.eval("files['mock/flybrain/csp-camera/oldreq/orphan-observation.json'].observer_generation")=='priorobserver'
    assert lua.eval("files['mock/flybrain/csp-camera/oldreq/result.json']") is None
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")=='prior generation unresolved; host review required'


def test_invalid_orphan_observation_fails_without_overwriting():
    lua,_=service(DEST)
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='oldreq',expected_generation='oldgen'}
      files['mock/flybrain/csp-camera/oldreq/stop']=''
      files['mock/flybrain/csp-camera/oldreq/orphan-observation.json']={actor_admissible=true}
      script.update(0)
    ''')
    assert lua.eval("files['mock/flybrain/csp-camera/oldreq/orphan-observation.json'].actor_admissible") is True
    assert lua.eval("files['mock/flybrain/csp-camera/oldreq/result.json']") is None
    assert 'differs' in lua.eval("files['mock/flybrain/csp-camera/status.json'].reason")


def test_release_write_failure_does_not_forge_witness():
    lua,_=service(DEST)
    lua.execute("failures['mock/flybrain/csp-camera/release-17_1.json']=1; releaseCallback()")
    assert lua.eval("files['mock/flybrain/csp-camera/release-17_1.json']") is None
