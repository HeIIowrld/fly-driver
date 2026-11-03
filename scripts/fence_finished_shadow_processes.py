"""Fence three exact completed shadow teardown processes, preserving their evidence."""
import ctypes,hashlib,json,os,subprocess,time
from ctypes import wintypes
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.win_process_receipt import ProcessReceipt
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
TARGETS=[(37108,'record_stationary_unified_file_shadow.py','20261101-stationary-unified-file-shadow'),(46632,'record_stationary_unified_multirate_shadow.py','20261101-post-restart-vision-shadow-1'),(25836,'record_stationary_unified_multirate_shadow.py','20261101-prefix8-freshness-defer-shadow-1')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 run=begin_run(B/'20261102-finished-shadow-fence-1',dict(program_sha256=sha(__file__),targets=TARGETS,scope='exact exit-requested completed owned children; preserve files; no UI or simulator control'));result={'status':'failed','processes':[],'camera_requests_sent':0,'lifecycle_requests_sent':0,'vehicle_control_sent':False};held=[]
 try:
  root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json';refuse_leased_transport('finished shadow teardown',status_path=root);st,mtime,_=read_session_heartbeat(root);validate_heartbeat(st,time.time()-mtime,mutating=False)
  if st.get('camera_active') is not False or st.get('ai_controlled') is not False or st.get('is_online') is not False or abs(st.get('speed_kmh',999))>=.1 or st.get('gas',1)>=.02 or st.get('brake',0)<=.98:raise ValueError('Observed idle stationary session required')
  for pid,script,name in TARGETS:
   folder=B/name;r=json.loads((folder/'result.json').read_text(encoding='utf8'));close=r.get('camera_close',{});term=close.get('producer_terminal',{});archive=r.get('archive',{})
   if r.get('status') not in ('stopped_on_error','stopped_on_guard_rejection') or r.get('vehicle_control_sent') is not False or close.get('status')!='stopped' or close.get('worker_alive') is not False or term.get('status')!='stopped' or term.get('request_id')!=r.get('camera_request_id') or archive.get('all_hashes_verified') is not True or archive.get('worker_alive') is not False or sha(folder/'vision/camera.zip')!=archive.get('bundle_sha256'):raise ValueError('Completed terminal and intact archive required: '+name)
   command=f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select-Object ProcessId,ExecutablePath,CommandLine,CreationDate | ConvertTo-Json -Compress";raw=subprocess.run(['pwsh','-NoLogo','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,encoding='utf8',check=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW);identity=json.loads(raw.stdout);cmd=identity.get('CommandLine','').replace('\\','/')
   if identity.get('ProcessId')!=pid or Path(identity.get('ExecutablePath','')).name.lower()!='python.exe' or script not in cmd or name not in cmd:raise ValueError('Exact owned process identity required')
   h=ProcessReceipt(pid);before=h.snapshot();held.append(h)
   if before['termination_signaled'] is not False or before['exit_requested'] is not True or before['exit_code_reported']!=0:raise ValueError('Process not in observed exit-requested zero-code state')
   result['processes'].append({'pid':pid,'identity':identity,'result_sha256':sha(folder/'result.json'),'archive_sha256':sha(folder/'vision/camera.zip'),'before':before,'termination_attempted':False})
  write_json(run/'preflight.json',{'session':st,'processes':result['processes']})
  api=ctypes.WinDLL('kernel32',use_last_error=True);api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE;api.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];api.TerminateProcess.restype=wintypes.BOOL;api.CloseHandle.argtypes=[wintypes.HANDLE]
  for row,h in zip(result['processes'],held,strict=True):
   if h.snapshot()['termination_signaled']:row['after']=h.snapshot();continue
   ph=api.OpenProcess(0x0001,False,row['pid'])
   if not ph:raise ctypes.WinError(ctypes.get_last_error())
   try:
    row['termination_attempted']=True;ok=api.TerminateProcess(ph,0);row['terminate_returned']=bool(ok);row['terminate_error']=0 if ok else ctypes.get_last_error()
   finally:api.CloseHandle(ph)
   deadline=time.monotonic()+3
   while time.monotonic()<deadline and not h.snapshot()['termination_signaled']:time.sleep(.05)
   row['after']=h.snapshot()
  result['status']='all_handles_signaled' if all(x.get('after',{}).get('termination_signaled') for x in result['processes']) else 'some_handles_still_active'
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  for h in held:h.close()
  write_json(run/'result.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
