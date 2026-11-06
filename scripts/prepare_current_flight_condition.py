"""Reconcile a stationary camera BEFORE the one-shot thermal condition in a new epoch."""
import argparse,os,re,subprocess,sys,time,json
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.preflight_current_flight_control import PIN
from scripts.audit_initial_thermal_evidence import audit
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat,target_reached,terminal_receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous-camera-evidence');p.add_argument('--calibration-result');p.add_argument('--observe-existing-restart',help='Exact accepted restart whose post-request heartbeat failed');p.add_argument('--temperature-c',type=int,choices=[40,80],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=None,ui_actions_sent=0));result=dict(status='failed',thermal_setter_command_invocations=0,actuator_messages_sent=0,ui_actions_sent=0);stages=[]
    def stage(name,module,args):
        with (out/(name+'.stdout.log')).open('x',encoding='utf8') as stdout,(out/(name+'.stderr.log')).open('x',encoding='utf8') as stderr:
            child=subprocess.Popen([sys.executable,'-m',module,*args,'--output',str(out/name)],cwd=Path.cwd(),stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write_json(out/'progress.json',dict(stage=name,pid=child.pid,started=time.perf_counter()))
            code=child.wait()
        stages.append(dict(stage=name,exitcode=code,original_handle=True))
        if code!=0:raise RuntimeError(name+' failed; original result retained')
    def lifecycle(name,command,extra=()):stage(name,'scripts.control_ac_session',['--command',command,'--expected-lua-sha256',PIN,'--execute',*extra])
    try:
        if a.observe_existing_restart:
            previous=Path(a.observe_existing_restart)
            failed,request,before=(read(previous/name) for name in ('result.json','request.json','before.json'))
            terminal=read(previous/'api-receipt.json')
            root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
            state,mtime,_=read_session_heartbeat(root/'status.json')
            validate_heartbeat(state,time.time()-mtime,mutating=True)
            if (failed['status']!='failed' or failed['lifecycle_requests_sent']!=1
                    or failed['reason']!='Fresh session API heartbeat required'
                    or request['command']!='restart' or terminal['status']!='api_returned_true'
                    or not re.fullmatch(r'[0-9a-f]{32}',request['id'])
                    or request['generation']!=before['generation'] or request['epoch']!=before['epoch']
                    or terminal_receipt(root,request)!=terminal or not target_reached('restart',before,state)
                    or state['epoch']!=before['epoch']+1 or state['window_handle']!=before['window_handle']
                    or state['track_id']!=before['track_id'] or state['car_id']!=before['car_id']
                    or state['camera_active'] or abs(state['speed_kmh'])>=.1
                    or state.get('tyres_initialization_epoch')==state['epoch']):
                raise ValueError('Exact accepted prior restart and fresh adjacent menu epoch required')
            write_json(out/'restart-observation.json',dict(source_result_sha256=sha256(previous/'result.json'),
                source_receipt_sha256=sha256(previous/'api-receipt.json'),request_id=request['id'],
                state=state,restart_requests_sent=0))
        else:
            lifecycle('restart','restart')
        lifecycle('start','start');lifecycle('pause','pause')
        if a.previous_camera_evidence:
            args=['--previous-evidence',a.previous_camera_evidence]
            if a.calibration_result:args+=['--calibration-result',a.calibration_result]
            stage('camera-reconciliation','scripts.reconcile_current_flight_camera',args)
        # No epoch restart or lengthy live diagnostic after this sole setter.
        result['thermal_setter_command_invocations']=1
        lifecycle('temperature','set-tyres-temperature-initial',['--live-initialization','--temperature-c',str(a.temperature_c),'--pause-after-readback','--axis-config','runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json','--resume-before-live-initialization','--settle-initial-thermal-readback'])
        proof=audit(out/'temperature');write_json(out/'thermal-audit.json',proof)
        result.update(status='completed_live_thermal_initial_condition',thermal_initialization_run=str(out/'temperature'),epoch=proof['epoch'],temperature_c=a.temperature_c,
            termination_reason='Camera-only diagnostic precedes one-shot temperature; no thermostat or second setter in epoch')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:
        try:
            state,mtime,_=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json');validate_heartbeat(state,time.time()-mtime,mutating=False)
            if not state['paused']:lifecycle('failure-pause','pause')
        except Exception as error:result['pause_error']=str(error)
        result['stages']=stages;write_json(out/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
