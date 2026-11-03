"""Bounded idle camera clock health observation; no capture or controls."""
import argparse,json,time
from pathlib import Path
from flybrain_drive.ac_csp_clock import CSPClockBounds,initialize_clock
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat


def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 root=Path.home()/'AppData/Local/flybrain';cam=root/'csp-camera';sessionp=root/'csp-session/status.json';run=begin_run(a.output,dict(program_sha256=sha256(__file__),camera_request=False,vehicle_control=False))
 result={'status':'failed','camera_request_sent':False,'vehicle_control_sent':False,'physical_exposure_sync_verified':False}
 try:
  before,modified,_=read_session_heartbeat(sessionp);validate_heartbeat(before,time.time()-modified,mutating=False)
  if (before.get('camera_active') is not False or before.get('ai_controlled') is not False or before.get('expert_probe_active') is not False or before.get('is_online') is not False or abs(before.get('speed_kmh',999))>=.1):raise ValueError('Idle stationary offline state required')
  camera_before=read(cam/'status.json');request=read(cam/'request.json');rid=request['id']
  if not rid.replace('-','').replace('_','').isalnum():raise ValueError('Safe request identifier required')
  terminal=read(cam/rid/'result.json')
  if terminal.get('request_id')!=rid or terminal.get('status') not in ('completed','failed','stopped') or camera_before.get('generation')!=before.get('generation'):raise ValueError('Same generation terminal evidence required')
  write_json(run/'before.json',{'session':before,'camera':camera_before,'request':request,'terminal':terminal})
  clock=CSPClockBounds(before['generation'],maximum_width=.1);initialize_clock(cam,clock,int(before['window_handle'].removesuffix('ULL')))
  after,modified,_=read_session_heartbeat(sessionp);validate_heartbeat(after,time.time()-modified,mutating=False);camera_after=read(cam/'status.json')
  if (after['generation']!=before['generation'] or after['camera_active'] is not False or after['ai_controlled'] is not False or after['expert_probe_active'] is not False or read(cam/'request.json')!=request or read(cam/rid/'result.json')!=terminal):raise ValueError('Session or terminal changed during observation')
  write_json(run/'after.json',{'session':after,'camera':camera_after,'clock':clock.record()})
  result.update(status='clock_exchange_observed',pings=len(clock.pings),clock_uncertainty_seconds=clock.high-clock.low,prior_terminal=terminal['status'],camera_before=camera_before.get('state'),camera_after=camera_after.get('state'),camera_error_after=camera_after.get('reason'),generation=after['generation'],session_heartbeat_failures_before=before.get('heartbeat_write_failures'),session_heartbeat_failures_after=after.get('heartbeat_write_failures'),interpretation='Clock reply writes work during this bounded idle observation; no evidence about the specific earlier failed write or camera capture health')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
