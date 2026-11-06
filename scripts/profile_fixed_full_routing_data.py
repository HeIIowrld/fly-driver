"""Measured input coverage of the existing episodes, not tyre limit capability."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.virtual_navigation_body import NAVIGATION_FEATURE_NAMES
from scripts.paired_decoder_data import load_data

ROOT=Path('runs/fixed-full-routing-learning-20261104')
def main():
 p=read('runs/multicorner-command-20261104/protocol-01.json');entries,items,_,_,pre=load_data(read(p['data_plan']));rows=[]
 for episode,(d,c,provenance) in enumerate(items):
  groups={}
  for role,mask in [('all',np.ones(len(d['body']),dtype=bool)),('fit',d['fit_mask']),('pre',pre[episode]&d['fit_mask'])]:
   stats={}
   for index in [0,1,2,3,4,5,6,7,12,13,14,15,16]:
    keep=mask&d['body_valid'][:,index];v=d['body'][keep,index];stats[NAVIGATION_FEATURE_NAMES[index]]=dict(valid=int(keep.sum()),minimum=None if not len(v) else float(v.min()),p05=None if not len(v) else float(np.quantile(v,.05)),p95=None if not len(v) else float(np.quantile(v,.95)),maximum=None if not len(v) else float(v.max()))
   groups[role]=dict(events=int(mask.sum()),features=stats)
  rows.append(dict(episode=episode,role='TRAIN' if episode<2 else 'consumed_same_course_development',episode_id=c['episode_id'],source=provenance,unique_camera_groups=len(np.unique(d['camera_index'])),groups=groups))
 write_json(ROOT/'data-coverage-01.json',dict(status='completed_existing_episode_input_coverage',program_sha256=sha256(__file__),episodes=rows,scope='Original observation values and explicit validity. Temperatures retain existing contract units/provenance. No friction utilization, limit handling, independent track, physical adaptation or performance is inferred from ranges.',actor_qualified=False,actual_tyre_adaptation_verified=False))
 print([{k:v for k,v in r.items() if k!='groups'} for r in rows])
if __name__=='__main__':main()
