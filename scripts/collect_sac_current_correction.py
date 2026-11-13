"""Owned completion collector for the declared correction replay and audit."""
import os,subprocess,time,json
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
ROOT=Path('runs/steering-precorners-20261103')
PYTHON=str(Path.home() / 'AppData/Local/PythonEnvs/flybrain-resolution-20261017/Scripts/python.exe')

def main():
 replay=Path('scripts/diagnose_sac_precorner_exchange_v2.py');audit=Path('scripts/audit_sac_visual_port_correction.py')
 hashes={str(p):sha256(p) for p in [replay,audit]}
 run=begin_run(ROOT/'correction-collection-01',dict(seed=780508,program_sha256=sha256(__file__),sealed_programs=hashes,owned_root=str(ROOT),timeout_seconds=1800,actor_deployed=False,vehicle_control_sent=False))
 jobs=[];start=time.monotonic();out=dict(status='failed',watcher_pid=os.getpid(),actor_deployed=False,vehicle_control_sent=False)
 try:
  fit=ROOT/'current-correction-fit-01'
  while not (fit/'result.json').exists():
   if time.monotonic()-start>1800:raise TimeoutError('Declared correction fit deadline reached')
   write_json(run/'progress.json',dict(phase='wait_fit',watcher_pid=os.getpid(),elapsed_seconds=time.monotonic()-start));time.sleep(5)
  r=json.loads((fit/'result.json').read_text(encoding='utf-8'))
  if r['status']!='completed_visual_port_correction_fit':raise ValueError('Correction fit failed; preserve result')
  for name,program,args in [('replay',replay,['--fit',str(fit),'--output',str(ROOT/'current-correction-stable-exchange-dev-01')]),('audit',audit,['--root',str(ROOT),'--output',str(ROOT/'correction-audit-01')])]:
   if sha256(program)!=hashes[str(program)]:raise ValueError('Sealed replay/audit source changed')
   target=Path(args[-1])
   if target.exists():raise FileExistsError('Do not overwrite '+str(target))
   cmd=[PYTHON,str(program),*args]
   with (run/(name+'-console.txt')).open('w',encoding='utf-8') as log:
    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    job=dict(name=name,pid=process.pid,command=cmd);jobs.append(job)
    while process.poll() is None:
     write_json(run/'progress.json',dict(phase=name,watcher_pid=os.getpid(),jobs=jobs,elapsed_seconds=time.monotonic()-start));time.sleep(5)
    job['exit_code']=process.returncode
   if (target/'result.json').exists():job['result_sha256']=sha256(target/'result.json')
   if process.returncode:raise ValueError('Declared '+name+' failed; log and result retained')
  out.update(status='completed_owned_correction_collection',termination_reason='Correction fit, declared same-body replay and independent evidence audit completed')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit collection failure; retain every artifact and child PID')
 out.update(jobs=jobs,elapsed_seconds=time.monotonic()-start);write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
