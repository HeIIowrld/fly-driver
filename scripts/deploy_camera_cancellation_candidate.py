"""Install tested camera cancel candidate on the exact stationary owned session.

No UI, lifecycle, camera request or actuator action. Installed bytes do not prove
that an already running Lua app reloaded. Original request and terminal untouched.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.build_camera_orphan_observation_candidate import build,DEST
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();source=Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua');target=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');root=Path(os.environ['LOCALAPPDATA'])/'flybrain';camera=root/'csp-camera';sid='mmf-stream-d487677e57a14987aea819bb4f516523';run=begin_run(a.output,dict(source_sha256=sha(source),candidate_sha256=sha(DEST),target=str(target),camera_request_id=sid,program_sha256=sha(__file__),ui_actions=0,actuator_actions=0));result={'status':'failed','installed':False,'producer_reload_verified':False,'lifecycle_requests_sent':0,'camera_requests_sent':0,'vehicle_control_sent':False};stage=None
 try:
  if DEST.read_bytes()!=build() or sha(target)!=sha(source):raise ValueError('Installed source or exact candidate differs')
  def preflight():
   refuse_leased_transport('camera cancellation candidate installation',status_path=root/'csp-session/status.json');st,mtime,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(st,time.time()-mtime,mutating=True);request=json.loads((camera/'request.json').read_text(encoding='utf8'));directory=camera/sid
   if (request.get('id')!=sid or request.get('expected_generation')!=st.get('generation') or not (directory/'stop').is_file() or (directory/'result.json').exists() or st.get('paused') is not False or st.get('live') is not True or st.get('camera_active') is not False or st.get('ai_controlled') is not False or st.get('is_online') is not False or st.get('in_menu') is not False or st.get('expert_probe_active') is not False or st.get('ideal_line_shown') is not False or abs(st.get('speed_kmh',999))>=.1 or st.get('gas',1)>=.02 or st.get('brake',0)<=.98):raise ValueError('Exact unresolved stopped request and original stationary session required')
   return st,request
  before,request=preflight();write_json(run/'before.json',dict(session=before,request=request,camera_status=json.loads((camera/'status.json').read_text(encoding='utf8'))));backup=run/'installed-original.lua';backup.write_bytes(target.read_bytes())
  if sha(backup)!=sha(source):raise ValueError('Backup mismatch')
  stage=target.with_name('FlyBrainCamera.lua.flybrain-cancel-stage')
  if stage.exists():raise ValueError('Existing staged file requires inspection')
  stage.write_bytes(DEST.read_bytes())
  if sha(stage)!=sha(DEST):raise ValueError('Candidate staging mismatch')
  fresh,again=preflight()
  if fresh['generation']!=before['generation'] or fresh['epoch']!=before['epoch'] or again!=request or sha(target)!=sha(source):raise ValueError('Preflight changed before installation')
  os.replace(stage,target);stage=None;result['installed']=True
  if sha(target)!=sha(DEST):raise ValueError('Installed hash mismatch')
  result.update(status='installed_reload_not_yet_verified',installed_sha256=sha(target),backup_sha256=sha(backup),generation=before['generation'])
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
