"""One source-preserving reload to observe the current Lua release callback."""
import argparse,hashlib,json,os,re,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.win_process_receipt import ProcessReceipt
from scripts.build_camera_release_witness_candidate import DEST,build
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();target=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');root=Path(os.environ['LOCALAPPDATA'])/'flybrain';camera=root/'csp-camera';sid='mmf-stream-d487677e57a14987aea819bb4f516523';run=begin_run(a.output,dict(program_sha256=sha(__file__),candidate_sha256=sha(DEST),request_id=sid,ui_actions=0));result={'status':'failed','replacement_done':False,'camera_requests_sent':0,'lifecycle_requests_sent':0,'vehicle_control_sent':False};handle=None
 try:
  if DEST.read_bytes()!=build() or sha(target)!=sha(DEST):raise ValueError('Exact installed candidate required')
  def preflight():
   refuse_leased_transport('release witness reload',status_path=root/'csp-session/status.json');st,mtime,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(st,time.time()-mtime,mutating=True);req=json.loads((camera/'request.json').read_text(encoding='utf8'));obs=json.loads((camera/sid/'orphan-observation.json').read_text(encoding='utf8'));cs=json.loads((camera/'status.json').read_text(encoding='utf8'))
   if (req.get('id')!=sid or obs.get('request_id')!=sid or obs.get('expected_generation')!=req.get('expected_generation') or obs.get('actor_admissible') is not False or cs.get('generation')!=st.get('generation') or cs.get('request_id')!=sid or (camera/sid/'result.json').exists() or not (camera/sid/'stop').is_file() or st.get('paused') is not False or st.get('live') is not True or st.get('camera_active') is not False or st.get('ai_controlled') is not False or st.get('is_online') is not False or st.get('expert_probe_active') is not False or st.get('ideal_line_shown') is not False or abs(st.get('speed_kmh',999))>=.1 or st.get('gas',1)>=.02 or st.get('brake',0)<=.98):raise ValueError('Exact unresolved orphan and stationary current producer required')
   return st,req,obs,cs
  before,req,obs,cs=preflight();handle=ProcessReceipt(34992);snap=handle.snapshot()
  if not snap['active']:raise ValueError('Exact old engine process must be alive at start')
  token=re.sub('[^a-zA-Z0-9_-]','_',before['generation']);witness=camera/('release-'+token+'.json')
  if witness.exists():raise ValueError('Current generation already has a release witness')
  write_json(run/'before.json',dict(session=before,request=req,orphan=obs,camera_status=cs,engine_handle=snap));stage=target.with_name('FlyBrainCamera.lua.flybrain-witness-reload-stage')
  if stage.exists():raise ValueError('Existing staged file requires inspection')
  stage.write_bytes(DEST.read_bytes())
  if sha(stage)!=sha(target):raise ValueError('Reload must use exactly installed bytes')
  fresh,req2,obs2,_=preflight()
  if fresh['generation']!=before['generation'] or fresh['epoch']!=before['epoch'] or req2!=req or obs2!=obs or sha(target)!=sha(DEST):raise ValueError('Session changed before replacement')
  os.replace(stage,target);result['replacement_done']=True;deadline=time.monotonic()+20
  while time.monotonic()<deadline:
   try:
    after,mtime,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(after,time.time()-mtime,mutating=False)
    if after['generation']!=before['generation'] and witness.exists():
     w=json.loads(witness.read_text(encoding='utf8'))
     if w.get('schema')!='flybrain.csp.camera.release_observation.v1' or w.get('generation')!=before['generation'] or w.get('actor_admissible') is not False:raise ValueError('Release witness identity differs')
     write_json(run/'after.json',dict(session=after,witness=w,engine_handle=handle.snapshot()));result.update(status='same_process_release_callback_observed',prior_generation=before['generation'],new_generation=after['generation'],release_witness=str(witness),engine_handle=handle.snapshot(),prior_request_terminal_present=(camera/sid/'result.json').exists());break
   except (FileNotFoundError,PermissionError,json.JSONDecodeError):pass
   time.sleep(.1)
  else:result.update(status='reload_or_release_unconfirmed',prior_generation=before['generation'],engine_handle=handle.snapshot())
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if handle:handle.close()
  write_json(run/'result.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
