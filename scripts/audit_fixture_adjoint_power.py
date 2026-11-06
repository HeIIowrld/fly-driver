"""Independently reconstruct speed policy and power current from recorded policy inputs."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/fixture-adjoint-learning-20261104')
def main():
 reports={}
 for arm in ['stop_all','input_adjoint']:
  fit=ROOT/f'{arm}-fit-01';path=ROOT/f'{arm}-evaluation-01';r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);n=read(ROOT/'normalization-01.json');assert cp['normalization_sha256']==sha256(ROOT/'normalization-01.json');assert torch.equal(cp['speed_policy']['reference'],torch.tensor(n['reference'],dtype=torch.float32)) and torch.equal(cp['speed_policy']['scale'],torch.tensor(n['scale'],dtype=torch.float32));w=cp['speed_policy']['projection.weight'].numpy().astype(float);b=cp['speed_policy']['projection.bias'].numpy().astype(float);rows=[];max_command=max_power=0.
  with np.load(path/'predictions.npz') as z:
   for episode in range(3):
    f=z[f'speed_features_{episode}'].astype(float);assert abs(f[:,:6]).max()<=5.;speed=f[:,-1]*30.;brain=f[:,:6]@w[0,:6];speed_logit=f[:,-1]*w[0,-1];correction=10*np.tanh(brain+speed_logit+b[0]);expected=np.minimum(np.maximum(speed+correction,0),3.*100/3.6);command=z[f'command_{episode}'][:,1];err=z[f'speed_error_{episode}'];power=z[f'power_{episode}'];pc=1/(1+np.exp(-(float(np.log(np.float32(.05)/(1-np.float32(.05))))+2*np.tanh(err/10.))));ce=float(abs(expected-command).max());pe=float(abs(pc[:,None]-power).max());max_command=max(max_command,ce);max_power=max(max_power,pe);rows.append(dict(episode=episode,events=len(f),correction_range_m_s=[float(correction.min()),float(correction.max())],observed_speed_error_range_m_s=[float(err.min()),float(err.max())],power_range=[float(power.min()),float(power.max())],brain_logit_range=[float(brain.min()),float(brain.max())],speed_logit_range=[float(speed_logit.min()),float(speed_logit.max())],encoded_symmetric_feature_peak_to_peak=np.ptp(f[:,:6],axis=0).tolist(),encoded_symmetric_feature_clip_fraction=(abs(f[:,:6])>=5).mean(0).tolist()))
  assert max_command<2e-5 and max_power<2e-7

  reports[arm]=dict(speed_weight=w.tolist(),speed_bias=b.tolist(),maximum_independent_command_error=max_command,maximum_independent_power_error=max_power,episodes=rows)
 write_json(ROOT/'power-route-audit-01.json',dict(status='completed_independent_speed_policy_and_power_route_audit',models=reports,actor_qualified=False,physical_feedback_stability_verified=False,vehicle_control_sent=False));print(reports)
if __name__=='__main__':main()
