"""Quarantine one exact unresolved stopped request without forging a terminal.

This removes only its publication slot by atomic same-volume rename; original
bytes and all per-request evidence remain. This is not admission for a new run.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.win_process_receipt import ProcessReceipt
from scripts.build_camera_release_witness_candidate import DEST,build
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(os.environ['LOCALAPPDATA'])/'flybrain';camera=root/'csp-camera';sid='mmf-stream-d487677e57a14987aea819bb4f516523';target=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');run=begin_run(a.output,dict(program_sha256=sha(__file__),candidate_sha256=sha(DEST),request_id=sid,ui_actions=0));result={'status':'failed','quarantined':False,'original_terminal_known':False,'new_camera_admitted':False,'camera_requests_sent':0,'lifecycle_requests_sent':0,'vehicle_control_sent':False};handle=None
 try:
  if DEST.read_bytes()!=build() or sha(target)!=sha(DEST):raise ValueError('Exact installed release-witness candidate required')
  def preflight():
   refuse_leased_transport('orphan request quarantine',status_path=root/'csp-session/status.json');st,mtime,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(st,time.time()-mtime,mutating=True);req=json.loads((camera/'request.json').read_text(encoding='utf8'));obs=json.loads((camera/sid/'orphan-observation.json').read_text(encoding='utf8'));cs=json.loads((camera/'status.json').read_text(encoding='utf8'))
   if (req.get('id')!=sid or obs.get('request_id')!=sid or obs.get('expected_generation')!=req.get('expected_generation') or obs.get('status')!='prior_generation_unresolved' or obs.get('actor_admissible') is not False or cs.get('generation')!=st.get('generation') or cs.get('request_id')!=sid or cs.get('reason')!='prior generation unresolved; host review required' or (camera/sid/'result.json').exists() or not (camera/sid/'stop').is_file() or st.get('paused') is not False or st.get('live') is not True or st.get('camera_active') is not False or st.get('ai_controlled') is not False or st.get('is_online') is not False or st.get('expert_probe_active') is not False or st.get('ideal_line_shown') is not False or abs(st.get('speed_kmh',999))>=.1 or st.get('gas',1)>=.02 or st.get('brake',0)<=.98):raise ValueError('Exact unresolved orphan and stationary producer required')
   return st,req,obs,cs
  source=camera/'request.json';destination=camera/sid/'quarantined-request.json'
  if destination.exists():raise ValueError('Existing quarantine requires review')
  st,req,obs,cs=preflight();handle=ProcessReceipt(34992)
  if not handle.snapshot()['active']:raise ValueError('Original engine handle must be live')
  query="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {$_.CommandLine -match 'record_stationary_unified_(mmf|file|multirate)_shadow.py'} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress";proc=subprocess.run(['pwsh','-NoLogo','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,encoding='utf8',check=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
  if proc.stdout.strip() not in ('','null','[]'):raise ValueError('A relevant shadow consumer process still exists')
  release=json.loads((camera/'release-19007968ULL_0_0016956.json').read_text(encoding='utf8'))
  if release.get('generation')!='19007968ULL:0.0016956' or release.get('active_before') is not False or release.get('active_after') is not False or release.get('prior_generation_resolved') is not False:raise ValueError('Exact prior release observation required')
  original_hash=sha(source);write_json(run/'before.json',dict(session=st,request=req,orphan=obs,camera_status=cs,prior_release=release,request_sha256=original_hash,engine_handle=handle.snapshot(),shadow_process_query='empty'));fresh,req2,obs2,_=preflight()
  if fresh['generation']!=st['generation'] or fresh['epoch']!=st['epoch'] or req2!=req or obs2!=obs or sha(source)!=original_hash or not handle.snapshot()['active']:raise ValueError('Ownership changed before quarantine')
  os.rename(source,destination);result['quarantined']=True
  if sha(destination)!=original_hash or source.exists():raise ValueError('Quarantine hash or empty publication slot check failed')
  deadline=time.monotonic()+5;observed=None
  while time.monotonic()<deadline:
   current,mtime,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(current,time.time()-mtime,mutating=False)
   if current['generation']!=st['generation'] or current['epoch']!=st['epoch'] or current.get('camera_active') is not False or current.get('paused') is not False or current.get('ai_controlled') is not False or source.exists() or (camera/sid/'result.json').exists() or not handle.snapshot()['active']:raise ValueError('State changed after quarantine; stop and inspect')
   if current['script_clock']>fresh['script_clock']+1.:observed=current;break
   time.sleep(.1)
  if observed is None:raise TimeoutError('No advancing same-producer heartbeat; quarantine outcome retained')
  write_json(run/'after.json',dict(session=observed,engine_handle=handle.snapshot(),quarantine_sha256=sha(destination),original_terminal_present=False));result.update(status='request_slot_quarantined_outcome_still_unknown',quarantine_path=str(destination),request_sha256=original_hash,generation=st['generation'],engine_handle=handle.snapshot())
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if handle:handle.close()
  write_json(run/'result.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
