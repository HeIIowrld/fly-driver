"""Finite paused camera/source timing observation; no vehicle or lifecycle writes."""
import argparse,json,os,threading,time
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_tcp import TcpTelemetryClient,record_packet
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
 source='16623d7a3226c9309d685d6497078064b4f181e43f1a972be9ddfb4d0a3ec1d9'
 axis=json.loads(Path('runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json').read_text(encoding='utf8'))
 sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256']);windows=WindowsACWindow();matches=windows.discover()
 if len(matches)!=1 or windows.user.GetForegroundWindow()==matches[0].hwnd:raise ValueError('One background original engine required')
 def idle():
  hb,mtime=read_published_json(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
  with NativePhysicsReader(timeout=.15) as native:sample=native.receive()
  if (not 0<=time.time()-mtime<=2 or not hb['paused'] or hb['camera_active'] or hb['ai_controlled'] or hb['is_online'] or hb['in_menu']
      or hb['window_handle']!=str(matches[0].hwnd)+'ULL' or sdk.GetVJDStatus(axis['device_id'])!=1
      or abs(sample.fields['speedKmh'])>=.1 or sample.fields['gas']>=.02 or sample.fields['brake']<=.98):raise ValueError('Fresh paused stationary full-brake AI-off FREE state required')
  return hb
 before=idle();run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_sha256=source,seed=780209,actuator_messages_sent=0,ui_actions_sent=0))
 result=dict(status='failed',training_admissible=False,actuator_messages_sent=0,ui_actions_sent=0,student_optimizer_updates=0)
 camera=None;stop=threading.Event();ready=threading.Event();packets=[];errors=[]
 def receive():
  try:
   with TcpTelemetryClient('127.0.0.1',timeout=1,clock=time.perf_counter) as client:
    client.subscribe();ready.set()
    while not stop.is_set() and len(packets)<5000:packets.append(record_packet(client.receive()))
  except Exception as error:errors.append(dict(error=type(error).__name__,reason=str(error)));ready.set()
 thread=threading.Thread(target=receive,daemon=True);thread.start()
 try:
  if not ready.wait(5) or errors:raise RuntimeError('Original TCP observer did not subscribe')
  config=json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
  camera=CSPVisionRecorder(matches[0].hwnd,config,max_frames=550,period=.09,max_bytes=1024**3,
   image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,
   mmf_metadata=True,source_timing_v1=True,lifecycle_only_camera_status=True,preserve_producer_time=True,expected_source_sha256=source)
  deadline=time.perf_counter()+60
  while camera.thread.is_alive():
   camera.thread.join(.1)
   if errors or time.perf_counter()>deadline:raise RuntimeError('Finite original timing observation failed')
  result['camera']=camera.close()
  if camera.error or camera.count!=550:raise RuntimeError('Original camera timing incomplete: '+str(camera.error))
  deadline=time.perf_counter()+2
  while True:
   try:after=idle();break
   except ValueError:
    if time.perf_counter()>deadline:raise
    time.sleep(.05)
  if before['generation']!=after['generation']:raise ValueError('Engine changed during observation')
  result.update(status='completed_stationary_source_timing',session_after=after)
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 finally:
  if camera is not None and camera.thread.is_alive():result['cleanup']=camera.close()
  stop.set();thread.join(2)
  if thread.is_alive() or errors:result.update(status='failed',tcp_errors=errors,tcp_alive=thread.is_alive())
  with (run/'tcp-source-packets.jsonl').open('w',encoding='utf8') as f:
   for row in packets:f.write(json.dumps(row)+'\n')
  if camera is not None and not camera.thread.is_alive() and camera.triggered:result['archive']=camera.archive_to(run/'vision',bundle=True)
  result.update(tcp_packets=len(packets),tcp_source_sha256=sha256(run/'tcp-source-packets.jsonl'))
  write_json(run/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
