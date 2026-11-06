"""Independent TRAIN statistics and previous-muscle symmetric encoding audit."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data
ROOT=Path('runs/fixture-adjoint-learning-20261104')
def semantic(x):return np.concatenate([abs(x[:,1::2]-x[:,::2]),(x[:,1::2]+x[:,::2])/np.float32(2)],axis=1)
def main():
 p=read(ROOT/'plan-01.json');n=read(ROOT/'normalization-01.json');assert n==p['speed_normalization'] and sha256(ROOT/'normalization-01.json')==p['normalization_sha256'];_,items,_,_,_=load_data(read(read('runs/multicorner-command-20261104/protocol-01.json')['data_plan']));parent=Path(p['normalized_parent_evaluation']);pr=read(parent/'result.json');assert sha256(parent/'predictions.npz')==pr['predictions_sha256']
 with np.load(parent/'predictions.npz') as z:x=np.concatenate([semantic(z[f'speed_features_{ep}'][items[ep][0]['fit_mask'],:6]) for ep in [0,1]]).astype(float)
 ref=x.mean(0).astype(np.float32);scale=np.maximum(x.std(0),1e-5).astype(np.float32);np.testing.assert_array_equal(ref,np.array(n['reference'],dtype=np.float32));np.testing.assert_array_equal(scale,np.array(n['scale'],dtype=np.float32));assert len(x)==n['TRAIN_decisions']==5659 and n['TRAIN_episodes']==[0,1] and n['development_used'] is False;reports={}
 for arm in ['stop_all','input_adjoint']:
  path=ROOT/f'{arm}-evaluation-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];rows=[]
  with np.load(path/'predictions.npz') as z:
   for ep in range(3):
    mu=z[f'muscles_{ep}'];previous=np.concatenate([np.zeros((1,6),dtype=np.float32),mu[:-1]],axis=0);expected=np.clip((semantic(previous)-ref)/scale,-5,5);observed=z[f'speed_features_{ep}'][:,:6];error=float(abs(expected-observed).max());assert error<2e-6;rows.append(dict(episode=ep,events=len(mu),maximum_independent_symmetric_encoding_error=error,clip_fraction=(abs(expected)>=5).mean(0).tolist()))
  reports[arm]=rows
 write_json(ROOT/'encoding-audit-01.json',dict(status='completed_independent_symmetric_speed_encoding_audit',TRAIN_statistics_exact=True,development_used=False,models=reports,actor_qualified=False,vehicle_control_sent=False));print(reports)
if __name__=='__main__':main()
