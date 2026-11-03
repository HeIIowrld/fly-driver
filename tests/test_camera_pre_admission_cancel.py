from pathlib import Path
from test_ac_lua_session_receipts import service, handled_luajit_errors_are_not_python_fatal_signals  # noqa: F401
from scripts.build_camera_pre_admission_cancel_candidate import build, DEST, SOURCE, OLD, NEW


def test_candidate_is_narrow_and_never_deployed():
    text=DEST.read_text(encoding='utf8')
    assert DEST.read_bytes()==build()
    assert text.replace(NEW,OLD)==SOURCE.read_text(encoding='utf8')


def test_stopped_unaccepted_request_is_terminal_before_paused_admission():
    lua,_=service(DEST)
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='request1',expected_generation='17:1'}
      files['mock/flybrain/csp-camera/request1/stop']=''
      script.update(0); now=2; sim.isPaused=true; script.update(0)
    """)
    assert lua.eval("files['mock/flybrain/csp-camera/request1/result.json'].status")=='stopped'
    assert lua.eval("files['mock/flybrain/csp-camera/request1/result.json'].camera_started") is False
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].request_id")=='request1'
    assert lua.eval('calls')==0


def test_failed_terminal_write_does_not_consume_request():
    lua,_=service(DEST)
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='request1',expected_generation='17:1'}
      files['mock/flybrain/csp-camera/request1/stop']=''
      failures['mock/flybrain/csp-camera/request1/result.json']=1
      script.update(0)
    """)
    assert lua.eval("files['mock/flybrain/csp-camera/request1/result.json']") is None
    lua.execute('now=2; script.update(0)')
    assert lua.eval("files['mock/flybrain/csp-camera/request1/result.json'].status")=='stopped'
    assert lua.eval('calls')==0


def test_wrong_generation_cannot_create_a_new_stopped_receipt():
    lua,_=service(DEST)
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='request1',expected_generation='other',frames=1,period_seconds=.1,deadline_seconds=3,exposure=.35}
      files['mock/flybrain/csp-camera/request1/stop']=''
      script.update(0)
    """)
    assert lua.eval("files['mock/flybrain/csp-camera/request1/result.json']") is None
    assert lua.eval("files['mock/flybrain/csp-camera/status.json'].state")=='failed'


def test_existing_terminal_is_never_replaced():
    lua,_=service(DEST)
    lua.execute("""
      files['mock/flybrain/csp-session/request.json']=nil
      files['mock/flybrain/csp-camera/request.json']={id='request1',expected_generation='17:1',frames=1,period_seconds=.1,deadline_seconds=3,exposure=.35}
      files['mock/flybrain/csp-camera/request1/stop']=''
      files['mock/flybrain/csp-camera/request1/result.json']={status='completed',frames=3}
      script.update(0)
    """)
    assert lua.eval("files['mock/flybrain/csp-camera/request1/result.json'].status")=='completed'
    assert lua.eval("files['mock/flybrain/csp-camera/request1/result.json'].frames")==3


def test_observed_v3_stop_is_retried_by_original_but_closed_by_candidate():
    # Reproduce the outstanding request's protocol fields in an isolated LuaJIT
    # filesystem. This test never touches the real producer or its request.
    def install(lua):
        lua.execute('''
          files['mock/flybrain/csp-session/request.json']=nil
          car.isAIControlled=false
          local id='mmf-stream-'..string.rep('a',32)
          files['mock/flybrain/csp-camera/request.json']={id=id,
            frames=80, period_seconds=.04, deadline_seconds=8.2, exposure=.35,
            image_format='bmp', verify_png_reference=false, shared_transport=true,
            mmf_metadata=true, host_persistence_protocol='immutable_ram_v3',
            expected_generation='17:1', source_timing_v1=true}
          files['mock/flybrain/csp-camera/'..id..'/stop']=''
          mmfCalls=0
          ac.writeMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('unexpected MMF open') end
          ac.readMemoryMappedFile=function() mmfCalls=mmfCalls+1; error('unexpected MMF open') end
        ''')

    original, _ = service(SOURCE)
    install(original)
    original.execute('script.update(0); now=2; script.update(0)')
    assert original.eval("files['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/result.json']") is None
    assert 'starting paused' in original.eval("files['mock/flybrain/csp-camera/status.json'].reason")
    assert original.eval('mmfCalls') == 0

    candidate, _ = service(DEST)
    install(candidate)
    candidate.execute('script.update(0); now=2; script.update(0)')
    receipt = candidate.eval("files['mock/flybrain/csp-camera/mmf-stream-'..string.rep('a',32)..'/result.json']")
    assert receipt['status'] == 'stopped' and receipt['frames'] == 0
    assert receipt['camera_started'] is False and receipt['actor_stream_created'] is False
    assert candidate.eval('mmfCalls') == 0 and candidate.eval('calls') == 0
