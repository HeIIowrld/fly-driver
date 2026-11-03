"""Audit CPU/GPU unified circuit parity on reconstructed original camera/body evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();g=B/'20261101-unified-real-recording-gpu';c=B/'20261101-unified-real-recording-cpu';gr=read(g/'result.json');cr=read(c/'result.json');gf=g/'unified-replay.npz';cf=c/'unified-replay.npz'
 if (gr['status']!='completed_unified_real_recording_offline_replay' or cr['status']!=gr['status'] or sha256(gf)!=gr['output_sha256'] or sha256(cf)!=cr['output_sha256'] or gr['source_reconstruction_sha256']!=cr['source_reconstruction_sha256'] or gr['candidate_sha256']!=cr['candidate_sha256'] or gr['training_admissible'] is not False or cr['training_admissible'] is not False):raise ValueError('matching diagnostic sources required')
 with np.load(gf,allow_pickle=False) as x,np.load(cf,allow_pickle=False) as y:
  np.testing.assert_array_equal(x['decision'],y['decision']);delta=np.abs(x['intents']-y['intents']);state=np.abs(x['states']-y['states']);gpu=x['intents'].copy();gt=x['timings_seconds'].copy();ct=y['timings_seconds'].copy()
 if state.shape!=(365,9728) or delta.shape!=(365,2) or not np.isfinite(state).all() or not np.isfinite(delta).all():raise ValueError('complete finite replay required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),gpu_sha256=sha256(gf),cpu_sha256=sha256(cf),source_reconstruction_sha256=gr['source_reconstruction_sha256']))
 result=dict(status='completed_unified_real_recording_cpu_gpu_audit',decisions=len(gpu),state_shape=list(state.shape),state_max_abs=float(state.max()),intent_max_abs=delta.max(axis=0).tolist(),intent_over_2e6=int(np.any(delta>2e-6,axis=1).sum()),state_over_2e6=int(np.any(state>2e-6,axis=1).sum()),turn_positive=int((gpu[:,0]>0).sum()),turn_negative=int((gpu[:,0]<0).sum()),gpu_timing_median_ms=float(np.median(gt[1:])*1000),cpu_timing_median_ms=float(np.median(ct[1:])*1000),training_admissible=False,optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False,physical_exposure_sync_verified=False,interpretation='Previously recorded diagnostic original image groups and body receipts, CPU/GPU unified model comparison; not a control or biological qualification')
 write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
