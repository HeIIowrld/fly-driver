import json,time
from types import SimpleNamespace
import pytest
import scripts.control_ac_session as module

class BeforeNative(Exception):pass

def exercise(tmp_path,monkeypatch,states,*,imola=False, source=None, full_resume=False):
 monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
 expected='owned-camera';window=SimpleNamespace(hwnd=7,executable='C:/engine/acs.exe')
 base=dict(paused=True,is_online=False,in_menu=False,ai_controlled=False,
  expert_probe_active=False,pending_receipt_id=None,error=None,generation='generation',window_handle='7ULL',
  schema='flybrain.csp.session.status.v1',epoch=3,car_id='bmw_z4_gt3',track_id='imola',
  live=False,script_clock=10,speed_kmh=0,gas=0,brake=1)
 calls=[]
 def heartbeat(path):
  row=dict(base,**states[min(len(calls),len(states)-1)]);calls.append(row)
  return row,time.time()-.01,0
 monkeypatch.setattr(module,'read_session_heartbeat',heartbeat)
 monkeypatch.setattr(module,'read_published_json',lambda path:(dict(state='capturing',request_id=expected),time.time()-.01))
 api=SimpleNamespace(user=SimpleNamespace(GetForegroundWindow=lambda:9),probe=lambda *args,**kwargs:window)
 monkeypatch.setattr(module,'WindowsACWindow',lambda:api)
 monkeypatch.setattr(module,'sha256',lambda path:source if str(path).endswith('.lua') and source else 'hash')
 class Reader:
  def __init__(self,**kwargs):pass
  def __enter__(self):
   if not full_resume:raise BeforeNative()
   self.packet=1
   return self
  def receive(self):
   sample=SimpleNamespace(packet_id=self.packet,fields=dict(speedKmh=0,gas=0,brake=1),
                          prefix_bytes=bytes(module.PHYSICS_PREFIX.size))
   self.packet+=1
   return sample
  def __exit__(self,*args):pass
 monkeypatch.setattr(module,'NativePhysicsReader',Reader)
 if full_resume:
  (tmp_path/'flybrain/csp-session').mkdir(parents=True)
  monkeypatch.setattr(module,'physics_record',lambda sample:dict(packet_id=sample.packet_id))
  monkeypatch.setattr(module,'terminal_receipt',lambda *args:dict(status='api_returned_true'))
  status=SimpleNamespace(snapshot=lambda:dict(builtin_ai_controlled=False),close=lambda:None)
  monkeypatch.setattr(module,'NativeTeacherStatusReader',lambda:status)
 owner=SimpleNamespace(plan=dict(request_id=expected,generation='generation',source_sha256=source or 'hash'),check=lambda *args:None)
 axis=dict(device_id=1,profile_path='profile',profile_sha256='hash');sdk=SimpleNamespace(GetVJDStatus=lambda device:1)
 out=tmp_path/'resume'
 return lambda:module.resume_owned_camera_for_recording(owner,output=out,window=window,axis=axis,sdk=sdk,imola=imola),out,calls


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


def test_imola_resume_publishes_owned_request_and_requires_packet_advance(tmp_path, monkeypatch):
 owned=dict(camera_active=True,camera_request_id='owned-camera',
            capabilities=dict(imola_stationary_lifecycle_v1=True))
 fn,out,_=exercise(tmp_path,monkeypatch,[owned,owned,dict(owned,paused=False,live=True)],
                   imola=True,source=module.IMOLA_OWNED_MMF_SOURCE_SHA256,full_resume=True)
 assert fn()['live']
 request=json.loads((out/'request.json').read_text())
 assert request['expected_track_id']=='imola'
 assert request['owned_mmf_resume'] and request['camera_request_id']=='owned-camera'
 assert json.loads((out/'before.json').read_text())['native']['packet_id']==1
 assert json.loads((out/'after.json').read_text())['native']['packet_id']==2
 result=json.loads((out/'result.json').read_text())
 assert result['lifecycle_requests_sent']==1 and result['actuator_messages_sent']==0


def test_imola_rejects_unpinned_source_before_preflight(tmp_path, monkeypatch):
 fn,out,calls=exercise(tmp_path,monkeypatch,[],imola=True)
 with pytest.raises(ValueError,match='Exact Imola owned MMF source'):
  fn()
 assert not calls and not out.exists()
