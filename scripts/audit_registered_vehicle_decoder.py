"""Independent float64 NumPy redecoding, fresh image and seven speed probes."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.evidence import write_json
from scripts.paired_decoder_data import load_data

def main():
 root=Path('runs/paired-muscle-decoder-20261104');fit=root/'registered-fit-01';r=read(fit/'result.json');assert sha256(fit/'registered-head.pt')==r['checkpoint_sha256'];cp=torch.load(fit/'registered-head.pt',map_location='cpu',weights_only=True);s={k:v.numpy().astype(float) for k,v in cp['readout'].items()};bias=float(cp['steering_bias']);plan=read(root/'plan-01.json');_,items,muscles,_,_=load_data(plan)
 def decode(mu):
  x=mu.astype(float);d=x[...,1::2]-x[...,::2];c=.5*(x[...,1::2]+x[...,::2]);ctx=np.concatenate([abs(d[...,:2]),c[...,:2]],-1);logits=np.clip((ctx-s['context_reference'])/s['context_scale'],-8,8)@s['context_weight'].T+np.clip((c[...,2:3]-s['power_reference'])/s['power_scale'],-8,8)*s['power_gain']*np.array([1,-1])+s['long_bias'];return np.concatenate([np.tanh(np.clip(d/s['difference_scale'],-8,8)@s['steering_weight']+bias)[...,None],1/(1+np.exp(-logits))],-1)
 with np.load(fit/'predictions.npz') as z:errors=[float(abs(decode(muscles[e])-z[f'prediction_{e}']).max()) for e in range(3)]
 assert max(errors)<2e-5
 speed=[]
 for role,path in [('TRAIN',Path(plan['functional_probes'])),('development',root/'development-probes-01')]:
  rr=read(path/'result.json');assert sha256(path/'probes.npz')==rr['raw_probes_sha256'];n=4 if role=='TRAIN' else 3
  with np.load(path/'probes.npz') as z:
   for j in range(n):
    p=decode(z[f'case_{j}_step_0.01_muscles']);speed.append(dict(role=role,case=j,longitudinal_derivative=float(((p[1,:,1]-p[1,:,2]-p[0,:,1]+p[0,:,2])/.02).mean())))
 groups=read(root/'reserved-parent-01/groups.json');counts={k:0 for k in ['fine','motion_off_after_prefix','speed_off_after_prefix']};target=0
 with np.load(root/'reserved-parent-01/responses.npz') as z:
  for g in groups:
   for arm in counts:
    p=decode(z[f"group_{g['group']}_{arm}_muscles"]);margin=float(-np.mean(p[1,25:,0]-p[0,25:,0])/2);counts[arm]+=int(margin>.0001);target+=int(arm=='fine' and margin>=.002)
 write_json(root/'registered-audit-01.json',dict(status='completed_independent_numpy_registered_decoder_audit',numpy_maximum_redecode_error=errors,speed_probes=speed,reserved_direction_counts=counts,reserved_target_pairs=target,yaw_mean_direction_gate=all(v['mean_steering_derivative']<0 for v in r['yaw_probes']),yaw_minimum_gain_gate=all(v['mean_steering_derivative']<=-.005 for v in r['yaw_probes']),speed_direction_gate=all(v['longitudinal_derivative']>0 for v in speed),steering_pre_rmse_reduction_percent=[100*(1-v['metrics']['pre']['registered']['rmse'][0]/v['metrics']['pre']['parent']['rmse'][0]) for v in r['actual_recordings']],checkpoint_sha256=r['checkpoint_sha256'],actor_qualified=False,vehicle_control_sent=False))
 print(read(root/'registered-audit-01.json'))
if __name__=='__main__':main()
