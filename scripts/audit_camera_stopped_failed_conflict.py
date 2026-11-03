"""Read-only known stopped/failed camera conflict and fresh producer clock probe."""
import argparse,json,os,time
from pathlib import Path
from flybrain_drive.ac_csp_clock import CSPClockBounds,initialize_clock
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.stationary_camera_admission import validate_known_stopped_metadata_failure_probe
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run=begin_run(a.output,dict(program_sha256=sha256(__file__),role='one non-actuating original status and clock diagnostic'));result=dict(status='failed',camera_requests_sent=0,actuator_messages_sent=0)
 try:
  local=Path(os.environ['LOCALAPPDATA'])/'flybrain';root=local/'csp-camera';status_path=local/'csp-session/status.json';st,modified,_=read_session_heartbeat(status_path);validate_heartbeat(st,time.time()-modified,mutating=False);request=read(root/'request.json');terminal=read(root/request['id']/'result.json');camera=read(root/'status.json');write_json(run/'before.json',dict(session=st,request=request,terminal=terminal,camera=camera))
  bounds=CSPClockBounds(st['generation'],maximum_width=.01);attempts=[];observe=bounds.observe
  def observed(sent,received,producer_clock,generation):
   attempts.append(dict(sent=sent,received=received,producer_clock=producer_clock,generation=generation));write_json(run/'clock-attempts.json',attempts);return observe(sent,received,producer_clock,generation)
  bounds.observe=observed;initialize_clock(root,bounds,int(st['window_handle'].removesuffix('ULL')),host_clock=time.perf_counter);fresh,modified,_=read_session_heartbeat(status_path);validate_heartbeat(fresh,time.time()-modified,mutating=False);cam=read(root/'status.json');clock=bounds.record()
  if read(root/'request.json')!=request or read(root/request['id']/'result.json')!=terminal:raise ValueError('Previous original request or terminal changed')
  validate_known_stopped_metadata_failure_probe(fresh,request,cam,terminal,clock,now=time.perf_counter());write_json(run/'after.json',dict(session=fresh,request=request,terminal=terminal,camera=cam,clock=clock));result.update(status='completed_specific_conflict_diagnostic',request_id=request['id'],before_sha256=sha256(run/'before.json'),after_sha256=sha256(run/'after.json'),interpretation='Exact stopped terminal and failed heartbeat remain contradictory; fresh clock does not prove camera health or automatically authorize retry')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result))
 if result['status']!='completed_specific_conflict_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
