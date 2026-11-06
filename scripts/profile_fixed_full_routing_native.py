"""Native19 assessed scope alongside legacy21 masks, no physical capability claim."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.native_multisensory_extension import EXTENSION_NAMES
from scripts.fixed_full_routing_experiment import FixedFullRoutingExperiment
ROOT=Path('runs/fixed-full-routing-learning-20261104')

def main():
 torch.set_num_threads(1);e=FixedFullRoutingExperiment(read(ROOT/'plan-01.json'),train_body=False);rows=[]
 try:
  for episode,(d,c,_) in enumerate(e.items):
   x=e.native[episode].numpy();valid=e.valid[episode].numpy();groups={}
   for role,mask in [('all',np.ones(len(x),dtype=bool)),('fit',d['fit_mask']),('pre',e.pre[episode]&d['fit_mask'])]:
    stats={}
    for index,name in enumerate(EXTENSION_NAMES):
     keep=mask&valid[:,index];v=x[keep,index];stats[name]=dict(valid=int(keep.sum()),minimum=None if not len(v) else float(v.min()),p05=None if not len(v) else float(np.quantile(v,.05)),p95=None if not len(v) else float(np.quantile(v,.95)),maximum=None if not len(v) else float(v.max()))
    groups[role]=dict(events=int(mask.sum()),native_features=stats)
   rows.append(dict(episode=episode,role='TRAIN' if episode<2 else 'consumed_same_course_development',groups=groups))
  e.verify_fixed();write_json(ROOT/'data-coverage-native-01.json',dict(status='completed_existing_native19_coverage',program_sha256=sha256(__file__),episodes=rows,scope='Assessed v2 native19, explicit masks/envelope; legacy21 acceleration masks remain invalid. This is coverage of actual inputs, not acceleration registration outside scope, tyre dynamics, grip utilization, handling limit or adaptation performance.',actor_qualified=False,actual_tyre_adaptation_verified=False))
 finally:e.close()
if __name__=='__main__':main()
