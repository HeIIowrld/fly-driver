"""PyTorch cross-check of NumPy/QP fitted child; no neural weights changed."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.visual_dn_current_calibration import PowerMonotoneVehicleReadout

def main():
 torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');fit=root/'one-sided-refit-01';r=read(fit/'result.json');plan=read(root/'one-sided-fit-plan-01.json');assert sha256(fit/'one-sided-head.pt')==r['checkpoint_sha256'] and sha256('scripts/fit_one_sided_joint_decoder.py')==plan['program_sha256'];parent=torch.load(root/'rank4-fit-01/joint-dn-decoder.pt',map_location='cpu',weights_only=True);cp=torch.load(fit/'one-sided-head.pt',map_location='cpu',weights_only=True);old=PowerMonotoneVehicleReadout(parent['readout']);head=PowerMonotoneVehicleReadout(cp['readout']);assert torch.equal(old.training_reference,head.training_reference) and torch.equal(old.training_scale,head.training_scale);errors=[];pedal_errors=[];one_sided=[]
 with torch.no_grad(),np.load(root/'rank4-evaluation-01/predictions.npz') as z,np.load(fit/'predictions.npz') as saved:
  for e in range(3):
   mu=torch.tensor(z[f'muscles_{e}']);p=head(mu).numpy();errors.append(float(abs(p-saved[f'prediction_{e}']).max()));pedal_errors.append(float(abs(p[:,1:]-old(mu).numpy()[:,1:]).max()))
 with torch.no_grad(),np.load(root/'yaw-decomposition-01/muscle-probes.npz') as z:
  for case in r['command_probes']:
   key=f"{case['role']}_{case['case']}";mu=torch.tensor(z[key+'_muscles']);neutral=torch.tensor(z[key+'_neutral_muscles']);p=head(mu.reshape(-1,6)).reshape(2,-1,3);base=head(neutral);minus=p[0,:,0]-base[:,0];plus=p[1,:,0]-base[:,0];one_sided.append(dict(role=case['role'],case=case['case'],mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),mean_direction_passed=bool(minus.mean()>0 and plus.mean()<0)))
 assert max(errors)<2e-6 and max(pedal_errors)==0;write_json(fit/'run-metadata-clarification.json',dict(seed=None,deterministic_solver=True,reason='No stochastic optimizer or random initialization; source configuration and source revision are in manifest. SLSQP ignored optional Hessian callback; quadratic objective and analytic gradient were used.'));write_json(root/'one-sided-independent-audit-01.json',dict(status='completed_one_sided_decoder_independent_audit',maximum_pytorch_vs_numpy_errors=errors,maximum_pedal_difference_from_parent=pedal_errors,one_sided_probes=one_sided,mean_bilateral_direction_passed=sum(v['mean_direction_passed'] for v in one_sided),minimum_hessian_eigenvalue=r['minimum_hessian_eigenvalue'],float32_logit_constraint_violation=r['float32_constraint_violation'],reserved_texture_seed=781191,reserved_pixels_generated=False,refit_rejected_before_reserved_test=True,actor_qualified=False,vehicle_control_sent=False));print(errors,pedal_errors)
if __name__=='__main__':main()
