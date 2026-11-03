"""Install release-witness candidate only on the observed unresolved camera session."""
import argparse,hashlib,json,os,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.build_camera_orphan_observation_candidate import build as build_v2
from scripts.build_camera_release_witness_candidate import DEST,build
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();target=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');root=Path(os.environ['LOCALAPPDATA'])/'flybrain';camera=root/'csp-camera';sid='mmf-stream-d487677e57a14987aea819bb4f516523';run=begin_run(a.output,dict(program_sha256=sha(__file__),installed_v2_sha256=hashlib.sha256(build_v2()).hexdigest(),candidate_sha256=sha(DEST),camera_request_id=sid,ui_actions=0));result={'status':'failed','installed':False,'producer_reload_verified':False,'lifecycle_requests_sent':0,'camera_requests_sent':0,'vehicle_control_sent':False}
 try:
  if DEST.read_bytes()!=build() or sha(target)!=hashlib.sha256(build_v2()).hexdigest():raise ValueError('Exact installed v2 and generated v3 required')
  def preflight():
   refuse_leased_transport('release-witness installation',status_path=root/'csp-session/status.json');st,mtime,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(st,time.time()-mtime,mutating=True);request=json.loads((camera/'request.json').read_text(encoding='utf8'));obs=json.loads((camera/sid/'orphan-observation.json').read_text(encoding='utf8'));cs=json.loads((camera/'status.json').read_text(encoding='utf8'))
   if (request.get('id')!=sid or obs.get('request_id')!=sid or obs.get('expected_generation')!=request.get('expected_generation') or obs.get('observer_generation')!=st.get('generation') or cs.get('generation')!=st.get('generation') or cs.get('request_id')!=sid or obs.get('actor_admissible') is not False or (camera/sid/'result.json').exists() or not (camera/sid/'stop').is_file() or st.get('paused') is not False or st.get('live') is not True or st.get('camera_active') is not False or st.get('ai_controlled') is not False or st.get('is_online') is not False or st.get('expert_probe_active') is not False or st.get('ideal_line_shown') is not False or abs(st.get('speed_kmh',999))>=.1 or st.get('gas',1)>=.02 or st.get('brake',0)<=.98):raise ValueError('Exact unresolved orphan and stationary current producer required')
   return st,request,obs,cs
  before,req,obs,cs=preflight();write_json(run/'before.json',dict(session=before,request=req,orphan=obs,camera_status=cs));backup=run/'installed-v2.lua';backup.write_bytes(target.read_bytes());stage=target.with_name('FlyBrainCamera.lua.flybrain-release-stage')
  if stage.exists():raise ValueError('Existing staged file requires inspection')
  stage.write_bytes(DEST.read_bytes())
  if sha(stage)!=sha(DEST):raise ValueError('Staged bytes differ')
  fresh,req2,obs2,_=preflight()
  if fresh['generation']!=before['generation'] or fresh['epoch']!=before['epoch'] or req2!=req or obs2!=obs or sha(target)!=sha(backup):raise ValueError('Session changed before replacement')
  os.replace(stage,target);result['installed']=True
  if sha(target)!=sha(DEST):raise ValueError('Installed bytes differ')
  result.update(status='installed_reload_not_yet_verified',installed_sha256=sha(target),backup_sha256=sha(backup),prior_generation=before['generation'])
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
