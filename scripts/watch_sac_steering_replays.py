"""Owned offline orchestration: collect every steering fit's visual replay."""
import argparse,json,os,subprocess,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--python',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.root);script=Path('scripts/diagnose_sac_precorner_exchange.py');script_sha=sha256(script)
 run=begin_run(a.output,dict(arguments=vars(a),seed=780504,program_sha256=sha256(__file__),replay_program_sha256=script_sha,owned_root=str(root),arms=['control','descending','optical'],timeout_seconds=2700,actor_deployed=False,vehicle_control_sent=False))
 handles={};logs={};jobs={};start=time.monotonic();out=dict(status='failed',watcher_pid=os.getpid(),actor_deployed=False,vehicle_control_sent=False)
 try:
  while len(jobs)<3 or any('exit_code' not in x for x in jobs.values()):
   if time.monotonic()-start>2700:raise TimeoutError('Offline fit/replay collection deadline reached')
   for arm in ['control','descending','optical']:
    if arm not in jobs:
     fit=root/(arm+'-fit-01');result=fit/'result.json'
     if not result.exists():continue
     r=json.loads(result.read_text(encoding='utf-8'))
     if r['status']!='completed_fixed_flight_steering_focus':
      jobs[arm]=dict(exit_code=-1,fit_failed=r);continue
     if sha256(script)!=script_sha:raise ValueError('Replay source changed after orchestration sealing')
     target=root/(arm+'-stable-exchange-dev-01')
     if target.exists():raise FileExistsError('Do not overwrite replay '+str(target))
     cmd=[a.python,str(script),'--fit',str(fit),'--output',str(target)]
     log=(run/(arm+'-console.txt')).open('w',encoding='utf-8');logs[arm]=log
     process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
     handles[arm]=process;jobs[arm]=dict(pid=process.pid,command=cmd,output=str(target),fit_checkpoint_sha256=r['checkpoint_sha256'])
    elif arm in handles and 'exit_code' not in jobs[arm]:
     code=handles[arm].poll()
     if code is not None:
      jobs[arm]['exit_code']=code;logs[arm].close()
      path=Path(jobs[arm]['output'])/'result.json'
      if path.exists():jobs[arm]['result_sha256']=sha256(path)
   write_json(run/'progress.json',dict(watcher_pid=os.getpid(),jobs=jobs,elapsed_seconds=time.monotonic()-start))
   if len(jobs)==3 and all('exit_code' in x for x in jobs.values()):break
   time.sleep(5)
  if any(j['exit_code']!=0 for j in jobs.values()):raise ValueError('At least one fit/replay failed; every result retained')
  out.update(status='completed_owned_steering_replay_collection',termination_reason='All three declared fits collected and same-body visual replays exited successfully')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit orchestration failure; preserve all jobs/logs and audit child PIDs')
 out.update(jobs=jobs,elapsed_seconds=time.monotonic()-start);write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
