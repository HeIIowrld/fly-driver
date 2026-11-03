"""Run an owned synthetic FlyVis load alongside the offline unified CPU replay."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),worker_sha256=sha256('scripts/owned_flyvis_load.py'),benchmark_sha256=sha256('scripts/benchmark_unified_numba_timestamped.py'),scope='offline independently running synthetic FlyVis and frozen historical unified graph; no control'))
 stop=run/'stop';worker=None;result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False);creation=getattr(subprocess,'CREATE_NO_WINDOW',0)
 try:
  with (run/'worker.log').open('wb') as log:
   worker=subprocess.Popen([sys.executable,'scripts/owned_flyvis_load.py','--output',str(run/'visual'),'--stop',str(stop)],cwd=os.getcwd(),stdout=log,stderr=subprocess.STDOUT,creationflags=creation)
   deadline=time.monotonic()+90
   while not (run/'visual'/'ready.json').exists():
    if worker.poll() is not None:raise RuntimeError('visual worker exited before readiness')
    if time.monotonic()>deadline:raise TimeoutError('visual worker readiness timeout')
    time.sleep(.25)
   with (run/'benchmark.log').open('wb') as benchlog:
    benchmark=subprocess.run([sys.executable,'scripts/benchmark_unified_numba_timestamped.py','--device','cpu','--output',str(run/'circuit')],cwd=os.getcwd(),stdout=benchlog,stderr=subprocess.STDOUT,creationflags=creation,timeout=180)
   if benchmark.returncode!=0:raise RuntimeError(f'circuit benchmark exit {benchmark.returncode}')
   stop.touch();code=worker.wait(timeout=30)
   if code!=0:raise RuntimeError(f'visual worker exit {code}')
  vr=read(run/'visual'/'result.json');br=read(run/'circuit'/'result.json');vf=run/'visual'/'visual-timing.npz';bf=run/'circuit'/'benchmark.npz'
  if vr['status']!='completed_owned_flyvis_load' or sha256(vf)!=vr['timing_sha256'] or br['status']!='completed_unified_offline_timestamped_benchmark' or sha256(bf)!=br['output_sha256']:raise ValueError('child evidence mismatch')
  with np.load(vf,allow_pickle=False) as z:v=z['windows'].copy()
  with np.load(bf,allow_pickle=False) as z:c=z['windows'].copy();duration=z['elapsed_seconds'].copy()
  overlap=np.array([np.any((v[:,0]<end)&(v[:,1]>start)) for start,end in c]);
  if not overlap.any():raise ValueError('no measured visual overlap')
  result.update(status='completed_offline_visual_contention',visual_chunks=len(v),circuit_decisions=len(c),overlap_count=int(overlap.sum()),overlap_median_ms=float(np.median(duration[overlap])*1000),overlap_p95_ms=float(np.quantile(duration[overlap],.95)*1000),nonoverlap_count=int((~overlap).sum()),nonoverlap_median_ms=float(np.median(duration[~overlap])*1000) if (~overlap).any() else None,circuit_median_ms=br['median_ms'],circuit_p95_ms=br['p95_ms'],circuit_max_ms=br['max_ms'],hypothetical_host_pass_after_priming=br['hypothetical_host_pass_after_priming'],eligible=br['eligible'],reference_max_abs_error=br['reference_max_abs_error'],visual_timing_sha256=sha256(vf),circuit_timing_sha256=sha256(bf),optimizer_updates=0,training_admissible=False,physical_exposure_sync_verified=False,interpretation='Separate repeated synthetic retinal load and offline precomputed scene circuit; measured host overlap; no pipeline transfer, pacing, real camera or closed loop')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if worker is not None and worker.poll() is None:
   stop.touch()
   try:worker.wait(timeout=30)
   except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=10)
  result['owned_worker_exit_code']=None if worker is None else worker.returncode;write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
