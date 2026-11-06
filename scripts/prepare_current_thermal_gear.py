"""Receipt-bound existing thermal condition -> verified gear-only paused start."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.thermal_teacher_start import admit_thermal_teacher_start
from scripts.audit_initial_thermal_evidence import audit
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

AXIS='runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--own-resume-during-gear',action='store_true');p.add_argument('--thermal-initialization-run',required=True);p.add_argument('--expected-lua-sha256',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--calibration-result',default='runs/thermal-driving-20261103/steering-calibration-02/result.json');p.add_argument('--output',required=True);a=p.parse_args()
    if not a.execute:raise ValueError('Explicit reviewed gear-only preparation required')
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=None,student_vehicle_control_sent=False,ui_actions_sent=0))
    result=dict(status='failed',student_vehicle_control_sent=False,ui_actions_sent=0);stages=[]
    def stage(name,module,args):
        with (run/(name+'.log')).open('x',encoding='utf-8') as log:
            command=[sys.executable,'-m',module,*args,'--output',str(run/name)]
            proc=subprocess.Popen(command,cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write_json(run/'progress.json',dict(stage=name,pid=proc.pid,command=command))
            code=proc.wait()
        outcome=read(run/name/'result.json');stages.append(dict(stage=name,returncode=code,status=outcome['status']))
        if code!=0 or outcome['status'] not in ('completed','read_only'):raise ValueError('Gear preparation failed '+name+': '+str(outcome.get('reason')))
    def lifecycle(name,command):stage(name,'scripts.control_ac_session',['--command',command,'--expected-lua-sha256',a.expected_lua_sha256,'--execute'])
    def session():
        state,mtime,_=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json');validate_heartbeat(state,time.time()-mtime,mutating=False);return state
    reader=None
    try:
        proof=audit(a.thermal_initialization_run);reader=NativeTeacherPhysicsReader();sample=repeated_teacher_snapshot(reader.mapping)
        before=session();initial=admit_thermal_teacher_start(proof,before,sample.physics.fields['tyreCoreTemperature'],source_sha256=a.expected_lua_sha256)
        write_json(run/'before.json',dict(session=before,native=sample.authority_record(),thermal_admission=initial))
        if not a.own_resume_during_gear:lifecycle('resume','resume')
        gear_args=['--execute','--ensure-first-gear','--gear-up-button','0','--pause-after-first-gear','--expected-lua-sha256',a.expected_lua_sha256,'--axis-config',AXIS,'--calibration-result',a.calibration_result]
        if a.own_resume_during_gear:gear_args+=['--resume-before-first-gear','--thermal-initialization-run',a.thermal_initialization_run]
        stage('gear','scripts.probe_ac_automatic_launch',gear_args)
        current=session();sample=repeated_teacher_snapshot(reader.mapping)
        admission=admit_thermal_teacher_start(proof,current,sample.physics.fields['tyreCoreTemperature'],source_sha256=a.expected_lua_sha256)
        write_json(run/'after.json',dict(session=current,native=sample.authority_record(),thermal_admission=admission))
        result.update(status='completed_current_thermal_first_gear',thermal_initialization_run=a.thermal_initialization_run,target_c=proof['target_c'],epoch=proof['epoch'],
            termination_reason='Existing thermal receipt unchanged; actual stationary first gear verified and API paused; gear-only control is preparation, not student action')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:
        try:
            state=session()
            if not state['paused']:lifecycle('final-pause','pause')
        except Exception as error:result.update(status='failed',pause_error=str(error))
        if reader is not None:reader.close()
        result['stages']=stages;write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
