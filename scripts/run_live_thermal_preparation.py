"""Explicit new episode, stationary live one-shot thermal set, raw audit and pause."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_initial_thermal_evidence import audit
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

AXIS='runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wake-native',action='store_true');p.add_argument('--resume-inside-thermal',action='store_true');p.add_argument('--settle-after-thermal',action='store_true');p.add_argument('--expected-lua-sha256',required=True);p.add_argument('--temperature-c',type=int,choices=[40,80,110],required=True);p.add_argument('--execute',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    if not a.execute:raise ValueError('Explicit one-new-episode live preparation required')
    if a.wake_native and a.resume_inside_thermal:raise ValueError('Choose owned resume or native wake preparation, not both')
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=None,actuator_messages_sent=0,ui_actions_sent=0))
    result=dict(status='failed',actuator_messages_sent=0,ui_actions_sent=0,thermal_setter_command_invocations=0);stages=[]
    def command(name,args):
        with (run/(name+'.log')).open('x',encoding='utf8') as log:
            command=[sys.executable,'-m','scripts.control_ac_session',*args,'--expected-lua-sha256',a.expected_lua_sha256,'--execute','--output',str(run/name)]
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=Path.cwd(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write_json(run/'progress.json',dict(stage=name,pid=child.pid,command=command,started=time.perf_counter()))
            code=child.wait()
        outcome=read(run/name/'result.json');stages.append(dict(stage=name,returncode=code,status=outcome['status']))
        if code!=0:raise ValueError('Explicit stage failed '+name+': '+str(outcome.get('reason')))
        return outcome
    try:
        command('restart',['--command','restart']);command('start',['--command','start'])
        command('pause-before-initial-resume',['--command','pause']) if a.resume_inside_thermal else command('resume',['--command','resume'])
        result['thermal_setter_command_invocations']=1
        thermal_args=['--command','set-tyres-temperature-initial','--live-initialization','--temperature-c',str(a.temperature_c),'--pause-after-readback','--axis-config',AXIS]
        if a.wake_native:thermal_args.append('--wake-native-before-live-initialization')
        if a.resume_inside_thermal:thermal_args.append('--resume-before-live-initialization')
        if a.settle_after_thermal:thermal_args.append('--settle-initial-thermal-readback')
        command('temperature',thermal_args)
        proof=audit(run/'temperature');write_json(run/'thermal-audit.json',proof)
        result.update(status='completed_live_thermal_initial_condition',thermal_initialization_run=str(run/'temperature'),epoch=proof['epoch'],temperature_c=a.temperature_c,
            termination_reason='One stationary live setter, three distinct original native temperature packets, independent audit and pause; no actor/vehicle control')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:
        state,modified,_=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
        validate_heartbeat(state,time.time()-modified,mutating=False)
        if state['paused'] is False:
            try:command('final-pause',['--command','pause'])
            except Exception as error:result['pause_error']=str(error)
        result['stages']=stages;write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
