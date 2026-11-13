"""Own the exact GPU child handle and retain its exit independently of terminal UI."""
import argparse
import json
import hashlib
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from flybrain_drive.evidence import write_json


def write(path,value):
    write_json(path,value)


def wait_owned_child(child,prefix,metadata_errors):
    """A publication error must never abandon the actual child process handle."""
    while True:
        try:return child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                write(prefix.with_suffix('.status.json'),dict(pid=child.pid,launcher_pid=os.getpid(),
                    poll_returncode=child.poll(),observed_utc=datetime.now(timezone.utc).isoformat(),
                    observed_host_time=time.perf_counter(),source='Original owned subprocess handle; no terminal-result or CIM inference'))
            except OSError as error:metadata_errors.append(type(error).__name__+': '+str(error))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['candidate-plan','thermal-initialization-run','output','receipt-prefix']:p.add_argument('--'+key,required=True)
    p.add_argument('--expected-lua-sha256');p.add_argument('--teacher-lateral-calibration');p.add_argument('--flush-body-denormal',action='store_true');p.add_argument('--defer-body-gc',action='store_true');p.add_argument('--profile-visual',action='store_true');p.add_argument('--defer-visual-gc',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('Explicit owned finite simulation execution required')
    prefix=Path(a.receipt_prefix)
    if Path(a.output).exists() or prefix.with_suffix('.started.json').exists():raise ValueError('Fresh owned run and launcher receipt required')
    binary=str(Path.home() / 'AppData/Local/Programs/Python/Python312/python.exe')
    library=str(Path.home() / 'AppData/Local/flybrain-bio-20261029/flyvis-env/Lib/site-packages')
    env=os.environ.copy();env['PYTHONPATH']=library+';src;.';env['OPENBLAS_NUM_THREADS']='1'
    command=[binary,'-m','scripts.shadow_native_flight_teacher_motion','--candidate-plan',a.candidate_plan,
        '--thermal-initialization-run',a.thermal_initialization_run,'--execute','--output',a.output]
    if a.expected_lua_sha256:command+=['--expected-lua-sha256',a.expected_lua_sha256]
    if a.teacher_lateral_calibration:command+=['--teacher-lateral-calibration',a.teacher_lateral_calibration]
    if a.flush_body_denormal:command.append('--flush-body-denormal')
    if a.defer_body_gc:command.append('--defer-body-gc')
    if a.profile_visual:command.append('--profile-visual')
    if a.defer_visual_gc:command.append('--defer-visual-gc')
    with prefix.with_suffix('.stdout.log').open('x',encoding='utf8') as out,prefix.with_suffix('.stderr.log').open('x',encoding='utf8') as error:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdout=out,stderr=error,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        errors=[]
        try:write(prefix.with_suffix('.started.json'),dict(pid=child.pid,launcher_pid=os.getpid(),command=command,
            cwd=str(Path.cwd()),started_utc=datetime.now(timezone.utc).isoformat(),started_host_time=time.perf_counter(),
            binary=binary,gpu_library=library,launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),student_actuator_access=False))
        except OSError as error:errors.append(type(error).__name__+': '+str(error))
        code=wait_owned_child(child,prefix,errors)
        write(prefix.with_suffix('.exit.json'),dict(pid=child.pid,exitcode=code,forced_termination=False,
            ended_utc=datetime.now(timezone.utc).isoformat(),ended_host_time=time.perf_counter(),
            metadata_errors=errors,retained_child_handle_during_metadata_error=True,
            source='Return code from the original owned subprocess handle; not CIM inventory or a result-file inference'))
    raise SystemExit(code)


if __name__=='__main__':main()
