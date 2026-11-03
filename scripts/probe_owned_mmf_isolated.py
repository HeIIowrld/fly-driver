"""Bounded stationary MMF paused-start and owned API resume diagnostic."""
import argparse,json,os,time,uuid,sys
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder,read_json,NoFreshCSPFrame
from flybrain_drive.ac_lifecycle_commit import fresh_unpublished_request
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.frame_persistence_profile_v3 import load_buffered_frame_profile, BufferedFrameWriterFactory, verify_buffered_persistence_archive
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.paused_mmf_startup import PausedMMFStartupGate
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat,terminal_receipt
from scripts.isolated_camera_diagnostic import IsolatedCamera

PIN='78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 sys.setswitchinterval(.001)
 profile_path='configs/ac-frame-persistence-buffered32-20261030.json';profile=load_buffered_frame_profile(profile_path);factory=BufferedFrameWriterFactory(profile)
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run=begin_run(a.output,dict(program_sha256=sha256(__file__),producer_sha256=PIN,gate_sha256=sha256('src/flybrain_drive/paused_mmf_startup.py'),vision_sha256=sha256('src/flybrain_drive/ac_csp_vision.py'),role='stationary diagnostic only',optimizer_updates=0,actor=False,python_switch_interval=sys.getswitchinterval(),shared_copy_sha256=sha256('src/flybrain_drive/ac_shared_texture.py'),buffered_profile_sha256=sha256(profile_path),buffered_writer_sha256=sha256('src/flybrain_drive/frame_persistence_profile_v3.py'),isolated_owner_sha256=sha256('scripts/isolated_camera_diagnostic.py')));result={'status':'failed','camera_requests_sent':0,'lifecycle_requests_sent':0,'vehicle_control_sent':False,'actor_qualified':False,'physical_time_verified':False};camera=None;reader=None;events=[]
 try:
  local=Path(os.environ['LOCALAPPDATA'])/'flybrain';root=local/'csp-camera';status_path=local/'csp-session/status.json';refuse_leased_transport('owned MMF resume diagnostic',status_path=status_path);before,modified,_=read_session_heartbeat(status_path);validate_heartbeat(before,time.time()-modified,mutating=True);cs=read_json(root/'status.json');prev=read_json(root/'request.json');terminal=read_json(root/prev['id']/'result.json');gate=PausedMMFStartupGate(before,prev,cs,terminal)
  if before.get('capabilities',{}).get('owned_mmf_resume') is not True or time.time()-(root/'status.json').stat().st_mtime>2:raise ValueError('Current producer capability and fresh camera status required')
  api=WindowsACWindow();windows=api.discover()
  if len(windows)!=1 or windows[0].pid!=38464 or int(before['window_handle'].removesuffix('ULL'))!=windows[0].hwnd or api.user.GetForegroundWindow()==windows[0].hwnd:raise ValueError('Exact owned background engine required')
  axis=read('configs/ac-axis-probe.json');sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
  if sdk.GetVJDStatus(axis['device_id'])!=1:raise ValueError('Released virtual device required')
  reader=NativePhysicsReader(timeout=.15);first=reader.receive();f=first.fields
  if abs(f['speedKmh'])>=.1 or f['gas']>=.02 or f['brake']<=.98:raise ValueError('Native stationary brake hold required')
  config=read('configs/ac-csp-exterior-front-v3-retina.json');write_json(run/'before.json',dict(session=before,camera=cs,previous_request=prev,previous_terminal=terminal,native_first_packet=first.packet_id));camera=IsolatedCamera(windows[0].hwnd,config,PIN,profile_path);result['camera_request_id']=camera.request_id;camera.wait_ready();result['camera_requests_sent']=int(camera.triggered)
  deadline=time.monotonic()+2
  while True:
   st,mtime,_=read_session_heartbeat(status_path);validate_heartbeat(st,time.time()-mtime,mutating=True);cam=read_json(root/'status.json')
   if st.get('camera_active') is True and st.get('camera_request_id')==camera.request_id and cam.get('request_id')==camera.request_id and cam.get('state')=='capturing':break
   if camera.error or not camera.thread.is_alive():raise RuntimeError('Camera ended before owned ready')
   if time.monotonic()>deadline:raise TimeoutError('Owned camera ready evidence not observed')
   time.sleep(.01)
  gate.camera_ready(st,cam,camera.request_id);request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,command='resume',generation=st['generation'],epoch=st['epoch'],expires_clock=st['script_clock']+3,camera_request_id=camera.request_id,owned_mmf_resume=True);journal=local/'csp-session/intent-journal';journal.mkdir(exist_ok=True);write_json(journal/(request['id']+'.json'),dict(request=request,session=st,camera=cam,role='owned_mmf_stationary_diagnostic'));fresh,mtime,_=read_session_heartbeat(status_path);validate_heartbeat(fresh,time.time()-mtime,mutating=True)
  if fresh.get('camera_request_id')!=camera.request_id:raise ValueError('Owned camera changed before publication')
  request=fresh_unpublished_request(st,fresh,request,publication_started=False);write_json(run/'resume-request.json',request);write_json(local/'csp-session/request.json',request);result['lifecycle_requests_sent']=1;result['lifecycle_request_id']=request['id'];deadline=time.monotonic()+4;receipt=None
  while time.monotonic()<deadline:
   receipt=terminal_receipt(local/'csp-session',request)
   if receipt is not None:break
   time.sleep(.01)
  if receipt is None:raise TimeoutError('Owned resume outcome unknown; never retry')
  write_json(run/'resume-receipt.json',receipt)
  if receipt.get('status')!='api_returned_true':raise ValueError('Owned resume API refused or failed')
  deadline=time.monotonic()+1.;observations=[]
  while True:
   after,mtime,_=read_session_heartbeat(status_path);validate_heartbeat(after,time.time()-mtime,mutating=False);cam=read_json(root/'status.json');later=reader.receive()
   observation=dict(session=after,camera=cam,first_body_packet=first.packet_id,later_body_packet=later.packet_id)
   try:gate.resume_observed(after,cam,first_body_packet=first.packet_id,later_body_packet=later.packet_id)
   except ValueError as e:
    observation['gate_error']=str(e);observations.append(observation);write_json(run/'handoff-observations.json',observations)
    if camera.error or not camera.thread.is_alive() or time.monotonic()>=deadline:raise
    time.sleep(.01);continue
   observations.append(observation);write_json(run/'handoff-observations.json',observations);write_json(run/'handoff.json',observation);break
  end=time.monotonic()+3.;last=None
  while time.monotonic()<end:
   if camera.error:raise RuntimeError('Owned camera consumer error: '+str(camera.error))
   try:frame,selected,_=camera.wait_for_candidate(max_age=.15,timeout=.04)
   except NoFreshCSPFrame:continue
   if frame['capture_id']!=last:
    packet=reader.receive();events.append(dict(capture_id=frame['capture_id'],camera_received=frame['received'],selection=selected,body_packet=packet.packet_id,body_received=packet.received_monotonic_seconds));last=frame['capture_id']
  if len(events)<2:raise ValueError('At least two distinct live frame groups required')
  result.update(status='completed_stationary_owned_mmf_resume_diagnostic',groups=len(events),first_body_packet=first.packet_id,later_body_packet=later.packet_id,training_admissible=False,interpretation='Stationary no-actuator diagnostic only; camera/body physical exposure timing and autonomous driving remain unverified')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if reader is not None:reader.close()
  if camera is not None:
   result['camera_requests_sent']=int(camera.triggered)
   try:
    result['camera_close']=camera.close()
    close=result['camera_close'];terminal=close.get('producer_terminal')
    if (close.get('status')!='stopped' or close.get('worker_alive') is not False
        or not isinstance(terminal,dict) or terminal.get('request_id')!=camera.request_id
        or terminal.get('status') not in ('stopped','completed')):
     result['status']='failed';result['camera_close_validation']='Consumer or matching producer terminal was not clean'
    if result['status']=='completed_stationary_owned_mmf_resume_diagnostic':
     archive=camera.archive_to(run/'vision',bundle=True)
     result['archive_receipt']=archive['receipt']
     if archive['count']!=close['frames']:raise ValueError('Verified persisted group count differs from consumer count')
     result['verified_persisted_groups']=archive['count']
   except Exception as e:result['camera_close_error']=type(e).__name__+': '+str(e);result['status']='failed'
  if camera is not None:
   try:camera.shutdown()
   except Exception as e:result['owner_shutdown_error']=type(e).__name__+': '+str(e);result['status']='failed'
  write_json(run/'events.json',events);result['events_sha256']=sha256(run/'events.json');write_json(run/'result.json',result)
 print(json.dumps(result))
 if result['status']!='completed_stationary_owned_mmf_resume_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
