"""Seal TRAIN-only symmetric normalization and exact continued source/moments."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand
from scripts.paired_decoder_data import load_data
ROOT=Path('runs/symmetric-speed-learning-20261104')
def main():
 ROOT.mkdir(exist_ok=False);prior=read('runs/flight-speed-policy-learning-20261104/source-receipt-final-01.json');old=dict(prior['preserved_production_and_checkpoint_sha256']);old.update(prior['current_checkpoints_sha256']);sources=dict(prior['preserved_prior_experiment_source_sha256']);sources.update({f:h for f,h in prior['source_and_documents_sha256'].items() if f.startswith(('src/','scripts/','tests/'))});assert all(sha256(f)==h for f,h in old.items()) and all(sha256(f)==h for f,h in sources.items());write_json(ROOT/'initial-preservation-01.json',dict(source_revision=prior['source_revision'],preserved_production_and_checkpoint_sha256=old,preserved_prior_experiment_source_sha256=sources))
 p=read('runs/flight-speed-policy-learning-20261104/plan-01.json');path=Path('runs/flight-speed-policy-learning-20261104/neutral_control-evaluation-01');assert sha256(path/'predictions.npz')==read(path/'result.json')['predictions_sha256'];_,items,_,_,_=load_data(read(read('runs/multicorner-command-20261104/protocol-01.json')['data_plan']));values=[]
 with np.load(path/'predictions.npz') as z:
  for episode in [0,1]:
   raw=torch.tensor(z[f'speed_features_{episode}'][items[episode][0]['fit_mask'],:6]);values.append(SymmetricNormalizedFlightSpeedCommand.symmetric_features(raw).numpy())
 features=np.concatenate(values).astype(float);reference=features.mean(0).astype(np.float32);scale=np.maximum(features.std(0),1e-5).astype(np.float32);n=dict(feature_order=['abs_pair0','abs_pair1','abs_pair2','mean_pair0','mean_pair1','mean_pair2'],reference=reference.tolist(),scale=scale.tolist(),floor=1e-5,clip=5.,TRAIN_episodes=[0,1],TRAIN_decisions=len(features),development_used=False,source_predictions_sha256=sha256(path/'predictions.npz'));write_json(ROOT/'normalization-01.json',n)
 parent='runs/flight-speed-policy-learning-20261104/neutral_control-fit-01/ports.pt';adam='runs/flight-speed-policy-learning-20261104/neutral_control-fit-01/optimizer-state.pt';fr=read(Path(parent).parent/'result.json');assert sha256(parent)==fr['checkpoint_sha256'] and sha256(adam)==fr['optimizer_checkpoint_sha256']
 p.update(normalized_parent_checkpoint=parent,normalized_parent_checkpoint_sha256=sha256(parent),normalized_parent_optimizer=adam,normalized_parent_optimizer_sha256=sha256(adam),normalized_parent_evaluation=str(path).replace(chr(92),'/'),normalization_sha256=sha256(ROOT/'normalization-01.json'),speed_normalization=n,initial_optimizer_steps=[790,662],seed=781509,reserved_seed=915320215,reserved_plan='runs/flight-speed-policy-learning-20261104/next-reserved-plan-01.json',scope='Same latest neutral-control parent and Adam790/662. Both continue motion8192/native9246. New8 speed coefficients frozen0 or learned with TRAIN-only abs paired muscle difference3+pair mean3 normalized/clipped5 and body speed/30. No oracle, original cumulative bounds, exact chemical graph and signed head. Same schedule/objective; no vehicle authority.',source_sha256={f:sha256(f) for f in ['src/flybrain_drive/symmetric_normalized_flight_speed.py','scripts/symmetric_speed_experiment.py','scripts/fit_symmetric_speed.py','scripts/validate_symmetric_speed.py','scripts/prepare_symmetric_speed.py']});p['reserved_plan_sha256']=sha256(p['reserved_plan']);write_json(ROOT/'plan-01.json',p);print(len(old),len(sources),n)
if __name__=='__main__':main()
