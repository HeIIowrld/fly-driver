"""Read-only evidence fence after one owned engine restart; no request publication."""
import argparse,hashlib,json,os,re,time
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
from flybrain_drive.camera_orphan_fence import validate_orphan_fence
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.win_process_receipt import ProcessReceipt
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run=begin_run(a.output,dict(program_sha256=sha256(__file__),validator_sha256=sha256('src/flybrain_drive/camera_orphan_fence.py'),role='read-only diagnostic fence, no terminal inference'));result={'status':'failed','camera_requested':False,'actor_qualified':False,'vehicle_control_sent':False};handle=None
 try:
  root=Path(os.environ['LOCALAPPDATA'])/'flybrain';camera=root/'csp-camera';sid='mmf-stream-d487677e57a14987aea819bb4f516523';qdir=B/'20261102-camera-request-quarantine-1';rrdir=B/'20261102-quarantine-engine-restart-2';q=read(qdir/'result.json');rst=read(rrdir/'result.json');rr=read(rrdir/'request.json');after=read(rrdir/'after.json');original=Path(q['quarantine_path']);req=read(original);obs=read(camera/sid/'orphan-observation.json');release=read(camera/'release-19007968ULL_0_0016075.json')
  if sha256(original)!=q['request_sha256'] or req['id']!=sid or rst.get('request_id')!=rr.get('id') or after.get('original_process',{}).get('termination_signaled') is not True or after.get('original_process',{}).get('pid')!=rr.get('old_engine_pid'):raise ValueError('Exact preserved request and original handle record required')
  lua=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua')
  if sha256(lua)!=sha256('integrations/ac_csp/candidates/FlyBrainCamera-release-witness-v3.lua'):raise ValueError('Reviewed installed producer differs')
  refuse_leased_transport('camera orphan fence',status_path=root/'csp-session/status.json');st,modified,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(st,time.time()-modified,mutating=True);cs=read(camera/'status.json');api=WindowsACWindow();windows=api.discover()
  if len(windows)!=1 or windows[0].pid!=rst['new_engine_pid'] or int(st['window_handle'].removesuffix('ULL'))!=windows[0].hwnd or after['session']['generation']!=st['generation'] or time.time()-(camera/'status.json').stat().st_mtime>2:raise ValueError('Fresh original new engine and camera identity required')
  handle=ProcessReceipt(windows[0].pid)
  if not handle.snapshot()['active']:raise ValueError('New engine must remain active')
  gate=validate_orphan_fence(request=req,orphan=obs,quarantine=q,restart=rst,restart_request=rr,release=release,current_session=st,current_camera=cs,engine_pid=windows[0].pid,request_slot_exists=(camera/'request.json').exists(),old_terminal_exists=(camera/sid/'result.json').exists())
  with NativeTeacherPhysicsReader() as reader:
   sample=reader.receive();f=sample.physics.fields
   if sample.builtin_ai_controlled or abs(f['speedKmh'])>=.1 or f['gas']>=.02 or f['brake']<=.98:raise ValueError('Fresh native AI off and stationary brake hold required')
   write_json(run/'native.json',sample.authority_record())
  write_json(run/'sources.json',dict(quarantine_sha256=sha256(qdir/'result.json'),request_sha256=sha256(original),restart_sha256=sha256(rrdir/'result.json'),restart_request_sha256=sha256(rrdir/'request.json'),restart_after_sha256=sha256(rrdir/'after.json'),orphan_sha256=sha256(camera/sid/'orphan-observation.json'),release_sha256=sha256(camera/'release-19007968ULL_0_0016075.json'),session=st,camera=cs,engine_handle=handle.snapshot()))
  result.update(status='completed_orphan_fence_evidence',gate=gate,sources_sha256=sha256(run/'sources.json'),native_sha256=sha256(run/'native.json'))
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if handle:handle.close()
  write_json(run/'result.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
