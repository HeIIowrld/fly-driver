"""Read-only real local-profile/NAS-stop notification preflight; no controls."""
import argparse,json,os,time
from pathlib import Path
from flybrain_drive.control_guard_metadata_process import ControlGuardMetadataProcess
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
 axis=json.loads(Path('runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json').read_text(encoding='utf8'))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),seed=780211,profile_path=axis['profile_path'],profile_sha256=axis['profile_sha256'],actuator_messages_sent=0,ui_actions_sent=0))
 reader=None;result=dict(status='failed',actuator_messages_sent=0,ui_actions_sent=0,student_optimizer_updates=0)
 try:
  reader=ControlGuardMetadataProcess(heartbeat_path=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json',
   profile_path=axis['profile_path'],stop_path=run/'stop-probe')
  deadline=time.perf_counter()+5
  while time.perf_counter()<deadline:
   row=reader.snapshot()
   if row['profile_sha256']!=axis['profile_sha256'] or not row['profile_watch_unchanged']:raise ValueError('Actual unchanged profile baseline differs')
   time.sleep(.01)
  issued=time.perf_counter();(run/'stop-probe').touch(exist_ok=False);deadline=issued+1
  observed=False
  while time.perf_counter()<deadline:
   try:reader.snapshot()
   except RuntimeError as error:
    if str(error)!='Owned run stop requested':raise
    observed=True;break
   time.sleep(.005)
  if not observed:raise TimeoutError('Original NAS stop notification was not observed')
  result.update(status='completed_real_profile_and_nas_stop_notification_probe',stop_detection_ms=(time.perf_counter()-issued)*1000)
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 finally:
  if reader is not None:
   result['close']=reader.close()
   with (run/'metadata.jsonl').open('xb') as f:
    for payload in reader.records:f.write(payload+b'\n')
   result['metadata_sha256']=sha256(run/'metadata.jsonl')
   if result['close']['status']!='completed' or result['close']['forced_termination']:result['status']='failed'
  write_json(run/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
