"""Decoder-only reference: original preserved muscle trajectories, no neural updates."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data

def main():
 root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');data=read(plan['paired_data_plan']);cp=root/'warm-fit-01/warm-head.pt';assert sha256(cp)==read(cp.parent/'result.json')['checkpoint_sha256'];s={k:v.numpy().astype(float) for k,v in torch.load(cp,map_location='cpu',weights_only=True)['readout'].items()}
 def decode(mu):
  z=((mu.astype(float)-s['training_reference'])/s['training_scale'])@s['projection.weight'].T+s['projection.bias'];return np.concatenate([np.tanh(z[...,:1]),1/(1+np.exp(-z[...,1:]))],-1)
 _,_,muscles,_,_=load_data(data)
 with np.load(cp.parent/'predictions.npz') as z:errors=[float(abs(decode(muscles[e])-z[f'prediction_{e}']).max()) for e in range(3)]
 assert max(errors)<2e-5;probes=[]
 for role,path in [('TRAIN',Path(data['functional_probes'])),('development',Path('runs/paired-muscle-decoder-20261104/development-probes-01'))]:
  rr=read(path/'result.json');assert sha256(path/'probes.npz')==rr['raw_probes_sha256']
  with np.load(path/'probes.npz') as z:
   for j in range(4 if role=='TRAIN' else 3):
    p=decode(z[f'case_{j}_step_0.01_muscles']);probes.append(dict(role=role,case=j,speed_longitudinal_derivative=float(((p[1,:,1]-p[1,:,2])-(p[0,:,1]-p[0,:,2])).mean()/.02)))
 with np.load(Path(plan['yaw_probes'])/'probes.npz') as z:
  for v in read(Path(plan['yaw_probes'])/'result.json')['cases']:
   p=decode(z[f"{v['role']}_{v['case']}_muscles"]);next(q for q in probes if q['role']==v['role'] and q['case']==v['case'])['yaw_steering_derivative']=float((p[1,:,0]-p[0,:,0]).mean()/.02)
 parent=root/'reserved-parent-01';pr=read(parent/'result.json');assert sha256(parent/'responses.npz')==pr['responses_sha256'];counts={k:0 for k in ['fine','motion_off_after_prefix','speed_off_after_prefix']};targets=0
 with np.load(parent/'responses.npz') as z:
  for g in read(parent/'groups.json'):
   for arm in counts:
    p=decode(z[f"group_{g['group']}_{arm}_muscles"]);margin=float(-np.mean(p[1,25:,0]-p[0,25:,0])/2);counts[arm]+=int(margin>.0001);targets+=int(arm=='fine' and margin>=.002)
 write_json(root/'warm-independent-audit-01.json',dict(status='completed_decoder_only_reference_audit',maximum_numpy_prediction_errors=errors,command_probes=probes,reserved_direction_counts=counts,reserved_target_pairs=targets,actual_recordings=read(root/'warm-recording-metrics-01.json'),actor_qualified=False,vehicle_control_sent=False,neural_recurrence_replayed=False));print(dict(probes=probes,direction=counts))
if __name__=='__main__':main()
