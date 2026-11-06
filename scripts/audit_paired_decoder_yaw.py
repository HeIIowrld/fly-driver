"""Independent original/paired yaw-command sign audit on preserved probes."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.paired_vehicle_candidate import load_paired_head
from flybrain_drive.sac_unified_student import MuscleVehicleReadout

def main():
 torch.set_num_threads(1);root=Path('runs/paired-muscle-decoder-20261104');plan=read(root/'plan-01.json');source=root/'yaw-probes-01';r=read(source/'result.json');assert sha256(source/'probes.npz')==r['raw_probes_sha256']
 state=torch.load(Path(plan['parent_fine_fit'])/'motion-spatial.pt',map_location='cpu',weights_only=True)['readout'];print({k:tuple(v.shape) for k,v in state.items()})
 # Decode the original affine head directly, separately from its production class.
 def original(mu):
  x=(mu-state['training_reference'])/state['training_scale'];v=torch.nn.functional.linear(x,state['projection.weight'],state['projection.bias']);return torch.cat([v[:,:1].tanh(),v[:,1:].sigmoid()],1)
 heads={'parent':original}
 for arm in ['monotone','free']:heads[arm]=load_paired_head(root/f'{arm}-fit-01/paired-head.pt')[0]
 reports=[]
 with torch.no_grad(),np.load(source/'probes.npz') as z:
  for case in r['cases']:
   key=f"{case['role']}_{case['case']}";mu=z[key+'_muscles'];neutral=z[key+'_neutral'];row=dict(case=case,models={})
   for name,head in heads.items():
    pred=head(torch.tensor(mu.reshape(-1,6))).numpy().reshape(2,-1,3);base=head(torch.tensor(neutral)).numpy();minus=pred[0,:,0]-base[:,0];plus=pred[1,:,0]-base[:,0];row['models'][name]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),mean_central_derivative=float(((pred[1,:,0]-pred[0,:,0])/.02).mean()),correct_minus_fraction=float((minus>1e-7).mean()),correct_plus_fraction=float((plus< -1e-7).mean()))
   reports.append(row)
 write_json(root/'yaw-audit-01.json',dict(status='completed_yaw_command_sign_audit',source_sha256=sha256(source/'probes.npz'),reports=reports,expected_positive_left_yaw_steering_sign=-1,actor_qualified=False,vehicle_control_sent=False))
 print(reports)
if __name__=='__main__':main()
