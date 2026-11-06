"""TRAIN-only attainable port current ranges; no additional neural replay."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment

def main():
 torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');path=root/'rank4-fit-01/joint-dn-decoder.pt';assert sha256(path)==read(path.parent/'result.json')['checkpoint_sha256'];cp=torch.load(path,map_location='cpu',weights_only=True);e=JointDNDecoderExperiment(plan,4,cp['readout']);e.calibration.load_state_dict(cp['current_calibration']);x=torch.cat([e.extras[k][e.items[k][0]['fit_mask']] for k in [0,1]]);result={}
 try:
  with torch.no_grad():
   original=e.dn(x);actual=e.calibration(x,original);q=(original/.2).clamp(1e-6,1-1e-6).logit();baseline=q.sigmoid();low=(original+.2*((q-plan['maximum_logit_shift']).sigmoid()-baseline)).clamp(0,.2);high=(original+.2*((q+plan['maximum_logit_shift']).sigmoid()-baseline)).clamp(0,.2);z=((x-e.calibration.reference)/e.calibration.scale).clamp(-5,5);shift=plan['maximum_logit_shift']*e.calibration.projection(z).tanh();prob=(q+shift).sigmoid();sensitivity=.2*prob*(1-prob);singular=torch.linalg.svdvals(actual-original)
   result=dict(status='completed_train_port_operating_point_audit',training_points=len(x),checkpoint_sha256=sha256(path),original_current_mean=original.mean(0).tolist(),actual_current_mean=actual.mean(0).tolist(),original_below_1e_minus5_fraction=(original<1e-5).float().mean(0).tolist(),original_above_point19999_fraction=(original>.19999).float().mean(0).tolist(),actual_current_change_maximum=(actual-original).abs().max(0).values.tolist(),maximum_attainable_current_high=high.max(0).values.tolist(),median_attainable_range_width=(high-low).median(0).values.tolist(),median_current_derivative_per_logit_shift=sensitivity.median(0).values.tolist(),actual_calibration_output_singular_values=singular.tolist(),neural_recurrence_replayed=False,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='TRAIN current operating range under this chosen bounded engineering logit correction. A low derivative is not proof of physiological inactivity or the sole source of driving failure.')
  e.verify_fixed();write_json(root/'current-operating-point-audit-01.json',result);print(result)
 finally:e.close()
if __name__=='__main__':main()
