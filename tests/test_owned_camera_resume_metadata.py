import json,time
from types import SimpleNamespace
import pytest
import scripts.control_ac_session as module

class BeforeNative(Exception):pass

def exercise(tmp_path,monkeypatch,states):
 monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
 expected='owned-camera';window=SimpleNamespace(hwnd=7,executable='C:/engine/acs.exe')
 base=dict(paused=True,is_online=False,in_menu=False,ai_controlled=False,
  expert_probe_active=False,pending_receipt_id=None,error=None,generation='generation',window_handle='7ULL')
 calls=[]
 def heartbeat(path):
  row=dict(base,**states[min(len(calls),len(states)-1)]);calls.append(row)
  return row,time.time()-.01,0
 monkeypatch.setattr(module,'read_session_heartbeat',heartbeat)
 monkeypatch.setattr(module,'read_published_json',lambda path:(dict(state='capturing',request_id=expected),time.time()-.01))
 api=SimpleNamespace(user=SimpleNamespace(GetForegroundWindow=lambda:9),probe=lambda *args,**kwargs:window)
 monkeypatch.setattr(module,'WindowsACWindow',lambda:api)
 monkeypatch.setattr(module,'sha256',lambda path:'hash')
 class Reader:
  def __init__(self,**kwargs):pass
  def __enter__(self):raise BeforeNative()
  def __exit__(self,*args):pass
 monkeypatch.setattr(module,'NativePhysicsReader',Reader)
 owner=SimpleNamespace(plan=dict(request_id=expected,generation='generation',source_sha256='hash'),check=lambda *args:None)
 axis=dict(device_id=1,profile_path='profile',profile_sha256='hash');sdk=SimpleNamespace(GetVJDStatus=lambda device:1)
 out=tmp_path/'resume'
 return lambda:module.resume_owned_camera_for_recording(owner,output=out,window=window,axis=axis,sdk=sdk),out,calls


def test_real_camera_frames_wait_for_independent_session_heartbeat_before_any_resume(tmp_path,monkeypatch):
 fn,out,calls=exercise(tmp_path,monkeypatch,[dict(camera_active=False),dict(camera_active=True,camera_request_id='owned-camera')])
 with pytest.raises(BeforeNative):fn()
 rows=json.loads((out/'preflight-observations.json').read_text())
 assert len(rows)==2 and len(calls)==2
 assert not (out/'request.json').exists()


def test_other_camera_id_is_not_waited_into_an_owned_resume(tmp_path,monkeypatch):
 fn,out,calls=exercise(tmp_path,monkeypatch,[dict(camera_active=True,camera_request_id='other')])
 with pytest.raises(ValueError,match='metadata agreement'):fn()
 assert len(calls)==1 and not (out/'request.json').exists()
