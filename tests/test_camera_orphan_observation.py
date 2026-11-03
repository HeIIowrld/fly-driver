"""Mock-only tests: a different producer generation cannot settle old ownership."""
from test_ac_lua_session_receipts import service, handled_luajit_errors_are_not_python_fatal_signals  # noqa: F401
from scripts.build_camera_orphan_observation_candidate import DEST, build


def install(lua):
    lua.execute('''
      files['mock/flybrain/csp-session/request.json']=nil
      local id='mmf-stream-'..string.rep('a',32)
      files['mock/flybrain/csp-camera/request.json']={id=id,expected_generation='old-generation'}
      files['mock/flybrain/csp-camera/'..id..'/stop']=''
      mmfCalls=0
      ac.writeMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('unexpected MMF') end
      ac.readMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('unexpected MMF') end
    ''')


def test_candidate_matches_builder_and_records_observation_not_terminal():
    assert DEST.read_bytes() == build()
    lua,_=service(DEST);install(lua)
    lua.execute('script.update(0); now=2; script.update(0)')
    root="files['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/'"
    report=lua.eval(root+"..'orphan-observation.json']")
    assert report['status']=='prior_generation_unresolved'
    assert report['old_process_exit_verified'] is False and report['prior_frames']=='unknown'
    assert report['actor_admissible'] is False
    assert lua.eval(root+"..'result.json']") is None
    assert lua.eval('mmfCalls')==0 and lua.eval('calls')==0


def test_failed_observation_write_retries_without_terminal_or_side_effect():
    lua,_=service(DEST);install(lua)
    lua.execute('''
      failures['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/orphan-observation.json']=1
      script.update(0)
    ''')
    key="files['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/orphan-observation.json']"
    assert lua.eval(key) is None
    lua.execute('now=2; script.update(0)')
    assert lua.eval(key)['status']=='prior_generation_unresolved'
    assert lua.eval('mmfCalls')==0 and lua.eval('calls')==0


def test_existing_terminal_remains_untouched():
    lua,_=service(DEST);install(lua)
    lua.execute('''
      files['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/result.json']={status='completed',frames=5}
      script.update(0)
    ''')
    root="files['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/'"
    assert lua.eval(root+"..'result.json'].frames")==5
    assert lua.eval(root+"..'orphan-observation.json']") is None


def test_same_generation_retains_pre_admission_cancellation():
    lua,_=service(DEST);install(lua)
    lua.execute('''
      files['mock/flybrain/csp-camera/request.json'].expected_generation='17:1'
      script.update(0)
    ''')
    root="files['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/'"
    receipt=lua.eval(root+"..'result.json']")
    assert receipt['status']=='stopped' and receipt['camera_started'] is False
    assert lua.eval(root+"..'orphan-observation.json']") is None
    assert lua.eval('mmfCalls')==0 and lua.eval('calls')==0
