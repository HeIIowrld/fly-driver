"""Prepare one new40/80/110C teacher-shadow episode using API and verified gear-only input."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_initial_thermal_evidence import audit

PIN='b2227739fb11d6ccce2020e976b2e0b3626d41ecd7dd8defb42ea8105a0495c2'
AXIS='runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gear-before-thermal',action='store_true',help='Explicit initial first-gear/brake native freeze before one thermal setter');p.add_argument('--verify-native-initialization',action='store_true',help='Opt-in diagnostic only; sleeping/native nonpublication remains an explicit failure');p.add_argument('--temperature-c',type=int,choices=[40,80,110],default=80);p.add_argument('--execute',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    if not a.execute:raise ValueError('Explicit one-new-episode preparation required')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),seed=None,temperature_initial_c=a.temperature_c,
        setter_policy='One call in a newly restarted epoch; only receipt-bound read-only verification if needed',ui_actions_sent=0))
    out=dict(status='failed',ui_actions_sent=0,stages=[])
    def stage(name,module,arguments,allowed=('completed',)):
        destination=run/name
        with (run/(name+'.log')).open('x',encoding='utf8') as log:
            command=[sys.executable,'-m',module,*arguments,'--output',str(destination)]
            child=subprocess.Popen(command,cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write_json(run/'progress.json',dict(stage=name,pid=child.pid,command=command,started=time.perf_counter()))
            code=child.wait()
        result=read(destination/'result.json');out['stages'].append(dict(name=name,returncode=code,status=result['status']))
        if result['status'] not in allowed:raise ValueError('Preparation failed at '+name+': '+str(result.get('reason')))
        return result
    def lifecycle(name,command):return stage(name,'scripts.control_ac_session',['--command',command,'--execute','--expected-lua-sha256',PIN])
    try:
        lifecycle('restart','restart');lifecycle('start','start')
        if a.gear_before_thermal:
            if a.verify_native_initialization:raise ValueError('Choose one explicit pre-thermal diagnostic sequence')
            lifecycle('resume-before-initial-gear','resume')
            stage('initial-first-gear-freeze','scripts.probe_ac_automatic_launch',['--execute','--ensure-first-gear','--gear-up-button','0',
                '--pause-after-first-gear','--expected-lua-sha256',PIN,'--axis-config',AXIS,
                '--calibration-result','runs/thermal-driving-20261103/steering-calibration-02/result.json'])
        else:
            if a.verify_native_initialization:
                lifecycle('resume-before-native-initialization','resume')
                stage('native-initialization','scripts.verify_native_initialization',['--axis-config',AXIS,'--expected-lua-sha256',PIN],allowed=('completed_native_initialization_readiness',))
            lifecycle('pause-before-temperature','pause')
        tag=str(a.temperature_c)
        thermal=['--temperature-c',tag,'--resume-for-readback','--pause-after-readback','--axis-config',AXIS,'--expected-lua-sha256',PIN,'--execute']
        result=stage('initial'+tag,'scripts.control_ac_session',['--command','set-tyres-temperature-initial',*thermal],allowed=('completed_initial_thermal_native_readback','failed'))
        condition=run/('initial'+tag)
        if result['status']=='failed':
            # Not a setter retry: require the original call's known terminal,
            # then independently verify fresh native packets in the same epoch.
            receipt=read(condition/'api-receipt.json')
            if receipt['status']!='thermal_calls_returned_readback_required':raise ValueError('Thermal SDK outcome unknown; never retry')
            lifecycle('pause-after-initial-failure','pause')
            stage('verify'+tag,'scripts.control_ac_session',['--command','verify-initial-tyres-temperature','--initialization-run',str(condition),*thermal],allowed=('completed_initial_thermal_native_readback',))
            condition=run/('verify'+tag)
        verified=audit(condition);write_json(run/'thermal-condition-audit.json',verified)
        lifecycle('resume-for-gear','resume')
        stage('first-gear-freeze','scripts.probe_ac_automatic_launch',['--execute','--ensure-first-gear','--gear-up-button','0',
            '--pause-after-first-gear','--expected-lua-sha256',PIN,'--axis-config',AXIS,
            '--calibration-result','runs/thermal-driving-20261103/steering-calibration-02/result.json'])
        out.update(status=f'completed_new{tag}c_motion_preparation',temperature_initial_c=a.temperature_c,thermal_initialization_run=str(condition),
            thermal_audit_sha256=sha256(run/'thermal-condition-audit.json'),epoch=verified['epoch'],
            actor_qualified=False,termination_reason='New epoch, single thermal call with raw readback audit and stationary gear-only API freeze completed; no student driving')
    except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit preparation failure; no setter or lifecycle retry')
    write_json(run/'result.json',out)
    if out['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
