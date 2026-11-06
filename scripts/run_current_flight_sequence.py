"""Unique owned thermal/gear + moving-shadow or student sequence; no UI fallback."""
import argparse,os,subprocess,sys,time,json
from pathlib import Path
from datetime import datetime,timezone
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.preflight_current_flight_control import PIN


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor-kind',choices=['parent','dual_guard','launch_parent','launch_dual_guard'],required=True)
    p.add_argument('--planned-student-seconds',type=float,default=.8)
    p.add_argument('--temperature-c',type=int,choices=[40,80],required=True)
    p.add_argument('--completed-metadata-evidence');p.add_argument('--mode',choices=['moving','student'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=begin_run(a.output,dict(arguments=vars(a),seed=None,ui_actions_sent=0))
    results=[];result=dict(status='failed',student_requests=0,ui_actions_sent=0)
    def stage(name,module,arguments):
        command=[sys.executable,'-m',module,*arguments]
        with (out/(name+'.stdout.log')).open('x',encoding='utf8') as stdout,(out/(name+'.stderr.log')).open('x',encoding='utf8') as stderr:
            child=subprocess.Popen(command,cwd=Path.cwd(),stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write_json(out/(name+'.started.json'),dict(pid=child.pid,command=command,started=time.perf_counter()))
            while True:
                try:code=child.wait(timeout=1);break
                except subprocess.TimeoutExpired:write_json(out/'progress.json',dict(stage=name,pid=child.pid,observed=time.perf_counter(),poll=child.poll()))
            receipt=dict(stage=name,pid=child.pid,exitcode=code,original_handle=True,finished=time.perf_counter())
            results.append(receipt);write_json(out/(name+'.exit.json'),receipt)
        if code!=0:raise RuntimeError(name+' failed; original logs and result retained')
    try:
        prep=out/'prepare';gear=out/'gear';run=out/'run'
        prepare_args=['--temperature-c',str(a.temperature_c),'--output',str(prep)]
        if a.completed_metadata_evidence:prepare_args+=['--previous-camera-evidence',a.completed_metadata_evidence]
        stage('prepare','scripts.prepare_current_flight_condition',prepare_args)
        thermal=str(prep/'temperature')
        stage('gear','scripts.prepare_current_thermal_gear',['--own-resume-during-gear','--thermal-initialization-run',thermal,'--expected-lua-sha256',PIN,'--execute','--output',str(gear)])
        checkpoints=dict(parent='runs/fixture-adjoint-learning-20261104/input_adjoint-fit-01/ports.pt',dual_guard='runs/dual-reference-learning-20261105/dual_guard-fit-01/ports.pt',launch_parent='runs/student-launch-learning-20261105/parent-pedal-fit-02/pedals.pt',launch_dual_guard='runs/student-launch-learning-20261105/dual_guard-pedal-fit-02/pedals.pt')
        checkpoint=checkpoints[a.actor_kind]
        cp_sha=sha256(checkpoint)
        if a.mode=='moving':
            module='scripts.shadow_current_flight_control';args=['--current-checkpoint',checkpoint,'--current-checkpoint-sha256',cp_sha,'--flush-body-denormal','--defer-body-gc','--profile-visual','--defer-visual-gc']
        else:module='scripts.pilot_current_flight_control';args=['--checkpoint',checkpoint,'--checkpoint-sha256',cp_sha]+['--planned-student-seconds',str(a.planned_student_seconds)]
        args+=['--actor-kind',a.actor_kind,'--candidate-plan','runs/motion-live-integration-20261104/candidate-plan-03.json','--thermal-initialization-run',thermal,'--expected-lua-sha256',PIN,'--execute','--output',str(run)]
        stage('execution','scripts.run_owned_flight_control',['--module',module,'--receipt-prefix',str(out/'owned'),'--',*args])
        r=read(run/'result.json');result.update(status='completed_sequence',run_status=r['status'],student_requests=r.get('student_requests',0))
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
        if (out/'run/result.json').exists():
            failed=read(out/'run/result.json');result.update(student_requests=failed.get('student_requests',0),run_status=failed['status'],run_termination_reason=failed.get('termination_reason'))
    finally:
        result['stages']=results;write_json(out/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
