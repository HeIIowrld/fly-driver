"""One owned pause verification, no vehicle command or desktop input."""
import argparse
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.owned_pause_process import OwnedPauseProcess
from scripts.preflight_current_flight_control import inspect,PIN
CPU=str(Path.home() / 'AppData/Local/Programs/Python/Python312/python.exe')
LIB=str(Path.home() / 'AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=begin_run(a.output,dict(seed=None,ui_actions_sent=0,vehicle_control_sent=False));before=inspect();write_json(out/'before-state.json',before);worker=OwnedPauseProcess(out/'pause',read(out/'manifest.json'),PIN,CPU,LIB)
 try:
  response=worker.pause();terminal=worker.close();after=inspect();result=dict(status='completed_preloaded_verified_pause' if response['exitcode']==0 and after['session']['paused'] else 'failed',ready=worker.ready,response=response,terminal=terminal,after=after,ui_actions_sent=0,actuator_messages_sent=0)
 finally:
  if not worker.closed:worker.close()
 write_json(out/'result.json',result);print(result['status'])
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
