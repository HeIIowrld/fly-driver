"""Mock-only lifecycle test for explicitly owned paused MMF resume."""
import pytest
from test_ac_lua_session_receipts import service, handled_luajit_errors_are_not_python_fatal_signals  # noqa: F401
from scripts.build_owned_mmf_resume_candidate import DEST,build


def ready():
    lua,_=service(DEST)
    lua.execute('''
      sim.isPaused=true; sim.isLive=false; sim.isInMainMenu=false; car.isAIControlled=false
      ac.disposeMemoryMappedFile=function() end
      local id='mmf-stream-'..string.rep('a',32)
      owned={directory='mock/flybrain/csp-camera/'..id,request={id=id,mmf_metadata=true,
        shared_transport=true,host_persistence_protocol='immutable_ram_v3',frames=80},
        mmfAck={},mmfFrame={},frames=1,deadline=9,last_capture_clock=1,next=100,views={}}
      for i=1,80 do local n=debug.getupvalue(script.update,i); if not n then break end
        if n=='active' then debug.setupvalue(script.update,i,owned);break end end
      request=files['mock/flybrain/csp-session/request.json'];request.command='resume'
      request.camera_request_id=id;request.owned_mmf_resume=true
    ''')
    return lua


def test_valid_owned_resume_once_without_ui_or_actuator():
    assert DEST.read_bytes()==build()
    lua=ready();lua.execute('script.update(0); now=1.1; script.update(0)')
    assert lua.eval('calls')==1 and lua.eval('sim.isPaused') is False
    terminal=lua.eval("files['mock/flybrain/csp-session/'..string.rep('a',32)..'.terminal.json']")
    assert terminal['status']=='api_returned_true' and terminal['ui_actions_sent']==0 and terminal['actuator_messages_sent']==0


@pytest.mark.parametrize('mutation',[
 "request.camera_request_id='mmf-stream-wrong'",'request.owned_mmf_resume=nil',
 'owned.request.mmf_metadata=false','owned.request.shared_transport=false',
 "owned.request.host_persistence_protocol='other'",'owned.mmfAck=nil','owned.mmfFrame=nil',
 'owned.frames=0','owned.frames=79','owned.deadline=2','owned.last_capture_clock=.5',
 "files[owned.directory..'/stop']=''",'car.isAIControlled=true','car.speedKmh=1',
 'car.gas=.03','car.brake=.9','sim.isOnlineRace=true','sim.isInMainMenu=true',
 "request.command='pause'",'request.generation="other"'])
def test_invalid_owned_resume_never_calls_api(mutation):
    lua=ready();lua.execute(mutation+'; script.update(0)')
    assert lua.eval('calls')==0 and lua.eval('sim.isPaused') is True
