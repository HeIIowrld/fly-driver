"""One paused read-only CSP wheel geometry observation; never controls a vehicle."""
import argparse,json,os,time,uuid
from pathlib import Path
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,repeated_snapshot
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.wheel_geometry_diagnostic import analyze_wheel_geometry
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--expected-candidate-sha256',required=True);ap.add_argument('--bridge-test-id',type=int,required=True);a=ap.parse_args();local=Path(os.environ['LOCALAPPDATA'])/'flybrain';camera=local/'csp-camera';status_path=local/'csp-session/status.json';installed=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');run=begin_run(a.output,dict(program_sha256=sha256(__file__),expected_candidate=a.expected_candidate_sha256,actor=False));result=dict(status='failed',vehicle_control_sent=False,actor_deployed=False,optimizer_updates=0);request_id='wheel-geometry-'+uuid.uuid4().hex
 try:
  refuse_leased_transport('wheel-geometry-diagnostic')
  status,mtime,_=read_session_heartbeat(status_path);validate_heartbeat(status,time.time()-mtime,mutating=True)
  if status.get('paused') is not True or status.get('is_online') is not False or status.get('camera_active') or status.get('ai_controlled') or abs(status.get('speed_kmh',999))>=.1 or status.get('gas',1)>=.02 or status.get('brake',0)<=.98:raise ValueError('paused offline idle brake-held car required')
  if sha256(installed)!=a.expected_candidate_sha256:raise ValueError('candidate source mismatch')
  cs=json.loads((camera/'status.json').read_text(encoding='utf8'))
  if cs.get('generation')!=status['generation'] or cs.get('state')=='capturing':raise ValueError('camera generation or ownership mismatch')
  with ExistingNamedMapping('Local\\acpmf_physics',PHYSICS_PREFIX.size) as mapping:
   before=repeated_snapshot(mapping)
   request=dict(id=request_id,frames=1,period_seconds=.04,deadline_seconds=10,exposure=.35,image_format='png',wheel_geometry_diagnostic=True,bridge_transport_test_id=a.bridge_test_id,expected_generation=status['generation'])
   write_json(run/'request.json',request);write_json(camera/'request.json',request);dest=camera/request_id;deadline=time.monotonic()+12
   while not (dest/'result.json').exists():
    if time.monotonic()>=deadline:raise TimeoutError('camera terminal not observed; inspect before any restore')
    time.sleep(.05)
   terminal=json.loads((dest/'result.json').read_text(encoding='utf8'));frame_path=dest/'frame-0000.json';frame=json.loads(frame_path.read_text(encoding='utf8'));after=repeated_snapshot(mapping)
  if terminal.get('status')!='completed' or terminal.get('request_id')!=request_id or frame.get('request_id')!=request_id or frame.get('generation')!=status['generation'] or frame.get('aid_probe_only') is not True:raise ValueError('exact diagnostic frame required')
  bridge=frame['bridge_transport_diagnostic'];obs=bridge['observed'];
  if bridge['test_id']!=a.bridge_test_id or bridge['same_context_only'] is not True or bridge['physics_thread_verified'] is not False or bridge['actor_admissible'] is not False or obs['generation']!=73 or obs['sequence']!=11 or obs['wheel_load_native_order']!=[101.25,202.5,303.75,404.0] or obs['atomicity_verified'] is not False:raise ValueError('CSP bridge observation differs')
  diagnostic=frame['wheel_geometry_diagnostic'];geometry=analyze_wheel_geometry(diagnostic);csp=[w['load_candidate'] for w in diagnostic['wheels']];native=[float(x) for x in after.fields['wheelLoad']];errors=[abs(x-y) for x,y in zip(csp,native,strict=True)]
  result.update(status='completed_same_context_bridge_diagnostic',request_id=request_id,generation=status['generation'],frame_sha256=sha256(frame_path),frame_path=str(frame_path),bridge=bridge,geometry=geometry,csp_loads=csp,native_loads=native,native_before_packet=before.packet_id,native_after_packet=after.packet_id,load_abs_errors=errors,physical_time_verified=False,physical_wheel_geometry_verified=False,native_order_registered=False,training_admissible=False,interpretation='One stationary renderer/physics diagnostic, independent timing and load semantics not verified; no actor or sensor admission.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e),request_id=request_id)
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_same_context_bridge_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
