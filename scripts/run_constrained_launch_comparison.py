"""Own the remaining fixed0.35s parent/candidate comparison stages in serial."""
import json,os,subprocess,sys,time
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.preflight_current_flight_control import inspect

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 base=Path('runs/student-launch-learning-20261105');out=begin_run(base/'constrained-comparison-owned-01',dict(program_sha256=sha256(__file__),seed=None,planned_student_seconds=.35,ui_actions_sent=0,automatic_retry=False));previous=base/'linear-candidate40-short-01/run';first=read(previous/'result.json');assert read(previous/'manifest.json')['config']['limits']['planned_student_seconds']==.35;assert first['pause_returncode']==0 and first['actuator_terminal']['stationary_feedback_verified'] and first['actuator_terminal']['device_released'];rows=[]
 for actor,temp,name in [('launch_parent',80,'constrained-parent80-short-01'),('launch_dual_guard',80,'constrained-candidate80-short-01')]:
  current=base/name;command=[sys.executable,'-m','scripts.run_constrained_flight_sequence','--actor-kind',actor,'--temperature-c',str(temp),'--mode','student','--planned-student-seconds','.35','--completed-metadata-evidence',str(previous),'--output',str(current)]
  with (out/(name+'.stdout.log')).open('x',encoding='utf8') as stdout,(out/(name+'.stderr.log')).open('x',encoding='utf8') as stderr:
   child=subprocess.Popen(command,cwd=Path.cwd(),stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));started=time.perf_counter();write_json(out/'progress.json',dict(stage=name,pid=child.pid,started=started))
   while True:
    try:code=child.wait(timeout=1);break
    except subprocess.TimeoutExpired:write_json(out/'progress.json',dict(stage=name,pid=child.pid,started=started,observed=time.perf_counter(),poll=child.poll()))
  row=dict(name=name,pid=child.pid,exitcode=code,original_child_handle=True,run=str(current),sequence=read(current/'result.json'));rows.append(row);write_json(out/(name+'.exit.json'),row)
  # Re-observe actual external stop/free state before the next fresh epoch.
  try:proof=inspect();write_json(out/(name+'.post-state.json'),proof)
  except Exception as error:write_json(out/'result.json',dict(status='blocked_external_cleanup',runs=rows,error=str(error),ui_actions_sent=0));raise
  if (current/'run/result.json').exists():previous=current/'run'
 write_json(out/'result.json',dict(status='completed_fixed_short_comparison' if all(v['exitcode']==0 for v in rows) else 'completed_fixed_short_comparison_with_failures',runs=rows,ui_actions_sent=0))
if __name__=='__main__':main()
