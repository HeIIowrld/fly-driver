"""Retain original child handle and exit receipts for CPU/GPU control experiments."""
import argparse,os,subprocess,time,json
from pathlib import Path
from datetime import datetime,timezone
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from scripts.run_owned_current_flight_shadow import wait_owned_child


def main():
    p=argparse.ArgumentParser();p.add_argument('--module',choices=['scripts.shadow_current_flight_control','scripts.pilot_current_flight_control','scripts.pilot_mailbox_flight_control','scripts.audit_current_flight_control_pixels','scripts.evaluate_sac_teacher_curriculum'],required=True)
    p.add_argument('--receipt-prefix',required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    prefix=Path(a.receipt_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    if a.module == 'scripts.evaluate_sac_teacher_curriculum' and '--heldout-magione-geometry' not in arguments:
        raise ValueError('This owned teacher launch is restricted to Magione calibration')
    if prefix.with_suffix('.started.json').exists():raise ValueError('Unique launcher receipt required')
    binary='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe'
    library='C:/Users/research-user/AppData/Local/flybrain-bio-20261029/flyvis-env/Lib/site-packages'
    env=os.environ.copy();env['PYTHONPATH']=library+';src;.';env['OPENBLAS_NUM_THREADS']='1'
    command=[binary,'-m','scripts.flight_control_worker_entry','--module',a.module,'--worker-receipt',str(prefix.with_suffix('.worker-terminal.json')),'--',*arguments];errors=[]
    with prefix.with_suffix('.stdout.log').open('x',encoding='utf8') as out,prefix.with_suffix('.stderr.log').open('x',encoding='utf8') as err:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:write_json(prefix.with_suffix('.started.json'),dict(pid=child.pid,launcher_pid=os.getpid(),command=command,cwd=str(Path.cwd()),started_utc=datetime.now(timezone.utc).isoformat(),started_host_time=time.perf_counter(),launcher_sha256=sha256(__file__),ui_actions_sent=0))
        except OSError as error:errors.append(str(error))
        code=wait_owned_child(child,prefix,errors)
        write_json(prefix.with_suffix('.exit.json'),dict(pid=child.pid,exitcode=code,ended_utc=datetime.now(timezone.utc).isoformat(),metadata_errors=errors,original_child_handle=True,forced_termination=False))
    raise SystemExit(code)

if __name__=='__main__':main()
