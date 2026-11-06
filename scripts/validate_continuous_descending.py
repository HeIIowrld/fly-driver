"""Zero-update continuous training/gradient cuts and synthetic interruption parity."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.descending_intent_experiment import DescendingExperiment
from scripts.continuous_descending_experiment import detach_exact,real_window,isolated_synthetic

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);e=DescendingExperiment(read(a.plan));out=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(a.plan),scope=__doc__,vehicle_control_sent=False));e.model.reset_episode();before=e.dn.projection.weight.detach().clone();predictions=[];cuts=[];synthetic=[]
 def update(kind,task,parts,prediction):
  sum(parts).backward();e.dn.projection.zero_grad(set_to_none=True);cuts.append(detach_exact(e.model));return dict(kind=kind,optimizer_update=False)
 try:
  for begin in range(0,1024,32):
   parts,prediction=real_window(e,0,begin,begin+32,learn=True);predictions.append(prediction.detach().numpy());sum(parts).backward();e.dn.projection.zero_grad(set_to_none=True);cuts.append(detach_exact(e.model))
   if begin in (128,512):
    record,digest=isolated_synthetic(e,dict(source_slot=0,pair=0,temperature=80.,group=1),update);synthetic.append(digest)
  prediction=np.concatenate(predictions);difference=float(abs(prediction-e.anchors[0][:1024]).max());assert difference<2e-6 and torch.equal(before,e.dn.projection.weight);e.verify_fixed();np.savez_compressed(out/'predictions.npz',prediction=prediction);write_json(out/'result.json',dict(status='completed_zero_update_continuous_training_parity',real_events=1024,real_gradient_windows=32,synthetic_interruptions=2,maximum_prediction_difference=difference,detach_numeric_hashes_unchanged=True,synthetic_restore_hashes=synthetic,gradient_cuts=len(cuts),optimizer_updates=0,predictions_sha256=sha256(out/'predictions.npz'),actual_driving_improvement_verified=False,vehicle_control_sent=False));print(json.dumps(dict(events=1024,maximum_difference=difference,cuts=len(cuts))))
 finally:e.close()
if __name__=='__main__':main()
