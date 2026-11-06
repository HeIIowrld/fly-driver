"""One API resume/pause with bracketed native graphics/body snapshots; no setters."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from flybrain_drive.ac_native_clock import NativeGraphicsTimingReader
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-lua-sha256',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    if not a.execute:raise ValueError('Explicit diagnostic resume/pause required')
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=None,actor=False,temperature_calls=0,awake_calls=0,actuator_messages_sent=0,ui_actions_sent=0))
    status=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json';body=None;graphics=None;child=None;rows=[];stages=[]
    outcome=dict(status='failed',actor=False,temperature_calls=0,awake_calls=0,actuator_messages_sent=0,ui_actions_sent=0)
    def state():
        value,mtime,_=read_session_heartbeat(status);validate_heartbeat(value,time.time()-mtime,mutating=False)
        return value
    def launch(command):
        log=(run/(command+'.log')).open('x',encoding='utf-8')
        args=[sys.executable,'-m','scripts.control_ac_session','--command',command,'--expected-lua-sha256',a.expected_lua_sha256,'--execute','--output',str(run/command)]
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=Path.cwd(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_json(run/(command+'-child.json'),dict(pid=proc.pid,command=args,started=time.perf_counter()))
        return proc,log
    try:
        before=state();write_json(run/'before.json',before)
        if before['paused'] is not True or before['camera_active'] or before['ai_controlled'] or before['is_online'] or before['expert_probe_active']:
            raise ValueError('Paused offline idle source required')
        body=NativeTeacherPhysicsReader();graphics=NativeGraphicsTimingReader()
        initial=repeated_teacher_snapshot(body.mapping)
        f=initial.physics.fields
        if initial.builtin_ai_controlled or abs(f['speedKmh'])>=.1 or f['gas']>=.02 or f['brake']<=.98:raise ValueError('Native stationary full brake required')
        write_json(run/'native-before.json',initial.authority_record())
        child,log=launch('resume');deadline=time.perf_counter()+30;live_start=None
        try:
            while time.perf_counter()<deadline:
                prior=state();g0=graphics.snapshot();sample=repeated_teacher_snapshot(body.mapping);g1=graphics.snapshot();after=state()
                rows.append(dict(host_clock=time.perf_counter(),session_before=prior,graphics_before=g0.metadata(),native=sample.authority_record(),graphics_after=g1.metadata(),session_after=after))
                if after['generation']!=before['generation'] or after['epoch']!=before['epoch']:raise ValueError('Original source/epoch changed')
                f=sample.physics.fields
                if sample.builtin_ai_controlled or after['ai_controlled'] or abs(f['speedKmh'])>=.1 or f['gas']>=.02 or f['brake']<=.98:
                    raise ValueError('Diagnostic stationary guard failed; pause required')
                if not after['paused'] and live_start is None:live_start=time.perf_counter()
                if child.poll() is not None and live_start is not None and time.perf_counter()-live_start>=3:break
                time.sleep(.02)
            code=child.wait(timeout=15);stages.append(dict(command='resume',returncode=code))
            if code!=0 or live_start is None:raise ValueError('Resume outcome/live observation failed')
            live=[r for r in rows if not r['session_after']['paused']]
            outcome.update(status='completed_native_resume_progress_diagnostic',observations=len(rows),live_observations=len(live),
                live_distinct_graphics_packets=len({r['graphics_after']['packet_id'] for r in live}),
                live_distinct_physics_packets=len({r['native']['packet_id'] for r in live}),
                live_native_graphics_statuses=sorted({r['graphics_after']['status'] for r in live}),
                sdk_live_source_clock_range=[live[0]['session_after']['script_clock'],live[-1]['session_after']['script_clock']],
                termination_reason='Read-only graphics/native comparison during one owned API resume; no temperature/wake or vehicle control')
        finally:
            log.close()
    except Exception as error:outcome.update(error=type(error).__name__,reason=str(error),termination_reason='Explicit diagnostic guard or observation failure; raw samples retained')
    finally:
        write_json(run/'samples.json',rows)
        if child is not None and child.poll() is None:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:outcome['resume_child_exit_unknown']=child.pid
        if child is not None and child.poll() is not None and not any(row['command']=='resume' for row in stages):
            stages.append(dict(command='resume',returncode=child.returncode))
        try:
            after=state()
            if not after['paused'] and not after.get('pending_receipt_id') and (child is None or child.poll() is not None):
                pause,log=launch('pause')
                try:code=pause.wait(timeout=30);stages.append(dict(command='pause',returncode=code))
                finally:log.close()
                if code!=0:raise ValueError('Cleanup pause failed')
            final=state();write_json(run/'after.json',final);outcome['final_paused']=final['paused']
            if not final['paused']:raise ValueError('Final pause unverified')
        except Exception as error:outcome.update(status='failed',cleanup_error=str(error))
        if body is not None:body.close()
        if graphics is not None:graphics.close()
        outcome['stages']=stages;outcome['samples_sha256']=sha256(run/'samples.json');write_json(run/'result.json',outcome)
    if outcome['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
