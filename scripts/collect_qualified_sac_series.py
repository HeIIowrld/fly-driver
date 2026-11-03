"""Serial verified API preparations, finite teacher episodes and independent audits.

No desktop input, altered sensing budgets, or forced actor process termination.

Each target directory is fresh; all failed episodes and receipts are retained.

"""

import argparse,json,subprocess,sys,os

from pathlib import Path

from flybrain_drive.evidence import begin_run,write_json

from flybrain_drive.graph import sha256
from flybrain_drive.ac_published_state import read_published_json

SOURCE='a99ef883eb2a37ce40db66c482817c9781d5754eb35039065ec62bb138f663ed'

AXIS='runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json'

CALIBRATION='runs/teacher-synchronized-20261102/steering-calibration-01/result.json'

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def main():

 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count',type=int,default=2);p.add_argument('--attempts',type=int,default=4);p.add_argument('--calibration-result',default=CALIBRATION);p.add_argument('--axis-config',default=AXIS);p.add_argument('--source-timing-diagnostic',action='store_true');p.add_argument('--full-source-timing-diagnostic',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()

 if a.full_source_timing_diagnostic:a.source_timing_diagnostic=True
 if not 1<=a.count<=4 or not a.count<=a.attempts<=8:raise ValueError('Finite explicit series required')
 if a.source_timing_diagnostic and (a.count!=1 or a.attempts!=1):raise ValueError('One finite separate timing diagnostic required')

 run=begin_run(a.output,dict(arguments=vars(a),seed=780206,program_sha256=sha256(__file__),source_sha256=SOURCE))

 result=dict(status='incomplete',qualified=[],attempts=[],student_optimizer_updates=0,ui_actions_sent=0)

 def invoke(module,args,out):

  command=[sys.executable,'-m',module,*args,'--output',str(out)]

  subprocess.run(command,cwd=Path.cwd(),check=True)

  return read(out/'result.json')

 try:

  for n in range(a.attempts):

   if (run/'stop').exists():raise RuntimeError('Owned series stop before next episode')
   episode=run/f'episode-{n+1:02d}';episode.mkdir()
   session,_=read_published_json(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
   if session['paused']:
    r=invoke('scripts.control_ac_session',['--command','resume','--execute','--expected-lua-sha256',SOURCE],episode/'resume-before-restart')
    if r['status']!='completed':raise RuntimeError('Original paused episode not resumed before reset; no deferred restart')

   for command in ('restart','start'):

    r=invoke('scripts.control_ac_session',['--command',command,'--execute','--expected-lua-sha256',SOURCE],episode/command)

    if r['status']!='completed':raise RuntimeError('Owned lifecycle failed; do not retry or fall back to desktop')

   r=invoke('scripts.probe_ac_automatic_launch',['--ensure-first-gear','--execute','--gear-up-button','0','--axis-config',a.axis_config,'--calibration-result',a.calibration_result],episode/'first-gear')

   if r['status']!='completed':raise RuntimeError('Actual first gear/stationary contract failed')

   r=invoke('scripts.control_ac_session',['--command','pause','--execute','--expected-lua-sha256',SOURCE],episode/'pause')

   if r['status']!='completed':raise RuntimeError('Owned pause failed')

   r=invoke('scripts.collect_synchronized_sac_demonstration',['--axis-config',a.axis_config,'--calibration-result',a.calibration_result,'--expected-lua-sha256',SOURCE,'--execute','--lifecycle-only-camera-status','--mmf-stop-signal','--memory-only-receive-checks',*(['--full-source-timing-diagnostic'] if a.full_source_timing_diagnostic else ['--source-timing-diagnostic'] if a.source_timing_diagnostic else [])],episode/'demonstration')

   attempt=dict(recording=str(episode/'demonstration'),status=r['status']);result['attempts'].append(attempt)

   if r['status']=='collected_unqualified_demonstration' and (not a.source_timing_diagnostic or a.full_source_timing_diagnostic):

    qr=invoke('scripts.audit_synchronized_sac_demonstration',['--recording',str(episode/'demonstration')],episode/'qualification')

    attempt['qualification_status']=qr['status']

    if qr['training_admissible']:

     result['qualified'].append(dict(recording=str(episode/'demonstration'),qualification=str(episode/'qualification'),qualification_sha256=sha256(episode/'qualification/result.json')))

   write_json(run/'progress.json',result)

   if len(result['qualified'])>=a.count:break

  if len(result['qualified'])==a.count:result['status']='completed_qualified_episode_series'

 except Exception as error:result.update(status='failed',error=type(error).__name__,reason=str(error))

 write_json(run/'result.json',result)

 print(json.dumps(result))

 if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()

