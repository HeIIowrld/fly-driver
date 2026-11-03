"""Install narrow owned-MMF-resume candidate with completed current request."""
import argparse,json,os,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.build_camera_release_witness_candidate import DEST as V3
from scripts.build_owned_mmf_resume_candidate import DEST,build
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();local=Path(os.environ['LOCALAPPDATA'])/'flybrain';root=local/'csp-camera';target=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');run=begin_run(a.output,dict(program_sha256=sha256(__file__),v3_sha256=sha256(V3),v4_sha256=sha256(DEST),ui_actions=0));out={'status':'failed','installed':False,'reload_verified':False,'camera_requests_sent':0,'lifecycle_requests_sent':0,'vehicle_control_sent':False}
 try:
  if DEST.read_bytes()!=build() or sha256(target)!=sha256(V3):raise ValueError('Exact v3 installation and v4 source required')
  def preflight():
   refuse_leased_transport('owned MMF candidate install',status_path=local/'csp-session/status.json');st,mtime,_=read_session_heartbeat(local/'csp-session/status.json');validate_heartbeat(st,time.time()-mtime,mutating=True);req=read(root/'request.json');cs=read(root/'status.json');terminal=read(root/req['id']/'result.json')
   if (req['id']!='mmf-probe-073992947a7e4c9db7456f7f5e461180' or terminal.get('request_id')!=req['id'] or terminal.get('status')!='completed' or terminal.get('frames')!=8 or cs.get('request_id')!=req['id'] or cs.get('state')!='completed' or cs.get('generation')!=st.get('generation') or st.get('paused') is not False or st.get('live') is not True or st.get('in_menu') is not False or st.get('camera_active') is not False or st.get('ai_controlled') is not False or st.get('is_online') is not False or st.get('ideal_line_shown') is not False or st.get('expert_probe_active') is not False or abs(st.get('speed_kmh',999))>=.1 or st.get('gas',1)>=.02 or st.get('brake',0)<=.98):raise ValueError('Exact completed probe and stationary idle session required')
   return st,req,cs,terminal
  st,req,cs,term=preflight();write_json(run/'before.json',dict(session=st,request=req,camera=cs,terminal=term));backup=run/'installed-v3.lua';backup.write_bytes(target.read_bytes());stage=target.with_name('FlyBrainCamera.lua.flybrain-owned-resume-stage')
  if stage.exists():raise ValueError('Prior staged file needs inspection')
  stage.write_bytes(DEST.read_bytes());fresh,req2,_,term2=preflight()
  if fresh['generation']!=st['generation'] or fresh['epoch']!=st['epoch'] or req2!=req or term2!=term or sha256(target)!=sha256(backup) or sha256(stage)!=sha256(DEST):raise ValueError('Session or bytes changed before install')
  os.replace(stage,target);out['installed']=True
  if sha256(target)!=sha256(DEST):raise ValueError('Candidate install differs')
  out.update(status='installed_reload_unverified',installed_sha256=sha256(target),previous_generation=st['generation'])
 except Exception as e:out.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',out)
 print(json.dumps(out))
if __name__=='__main__':main()
