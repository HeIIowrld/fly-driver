"""Quantify recorded flight muscle scale on qualified fitting decisions only."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data
ROOT=Path('runs/flight-speed-policy-learning-20261104')
def main():
 _,items,_,_,_=load_data(read(read('runs/multicorner-command-20261104/protocol-01.json')['data_plan']));path=ROOT/'speed_policy-evaluation-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];cp=torch.load(ROOT/'speed_policy-fit-01/ports.pt',map_location='cpu',weights_only=True);w=cp['speed_policy']['projection.weight'].numpy()[0].astype(float);b=float(cp['speed_policy']['projection.bias'][0]);rows=[]
 with np.load(path/'predictions.npz') as z:
  for episode,(d,*_) in enumerate(items):
   f=z[f'speed_features_{episode}'][d['fit_mask']].astype(float);muscle_logit=f[:,:6]@w[:6];speed_logit=f[:,-1]*w[-1];fixed_speed_correction=10*np.tanh(muscle_logit+w[-1]+b);pairs=f[:,1:6:2]-f[:,:6:2];rows.append(dict(episode=episode,qualified_decisions=len(f),muscle_peak_to_peak=np.ptp(f[:,:6],axis=0).tolist(),muscle_std=f[:,:6].std(0).tolist(),pair_difference_peak_to_peak=np.ptp(pairs,axis=0).tolist(),pair_difference_std=pairs.std(0).tolist(),muscle_logit_range=[float(muscle_logit.min()),float(muscle_logit.max())],speed_logit_range=[float(speed_logit.min()),float(speed_logit.max())],fixed_speed30_m_s_correction_range=[float(fixed_speed_correction.min()),float(fixed_speed_correction.max())],fixed_speed30_m_s_muscle_modulation_peak_to_peak=float(np.ptp(fixed_speed_correction))))
 write_json(ROOT/'policy-scale-diagnostic-01.json',dict(status='completed_qualified_flight_muscle_speed_policy_scale_diagnostic',source_predictions_sha256=r['predictions_sha256'],episodes=rows,scope='Recorded own muscle states varied at fixed 30m/s in the existing stateless policy. Not an end-to-end image counterfactual, physical target-speed truth, causal attribution fraction or new-data generalization.',actor_qualified=False,vehicle_control_sent=False));print(rows)
if __name__=='__main__':main()
