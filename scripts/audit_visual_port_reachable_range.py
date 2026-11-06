"""Analytic reachable DN opponent-current intervals, not a vehicle controllability proof."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.visual_descending_port import VisualDescendingPort
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read

ROOT=Path('runs/visual-native-port-learning-20261104')
def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');oldplan=read(plan['parent_fit_plan']);original=torch.load('runs/motion-spatial-learning-20261104/fine_motion-fit-01/motion-spatial.pt',map_location='cpu',weights_only=True)['adapter'];dn=VisualDescendingPort(original['reference'],original['scale'],original['orientations']);dn.load_state_dict(original);source=ROOT/'reserved-stimuli-01';r=read(source/'result.json');assert sha256(source/'features.npz')==r['features_sha256']
 with np.load(source/'features.npz') as z:x=torch.tensor(z['static'][:,25:].reshape(-1,320))
 with torch.no_grad():
  base=dn(x);maximum=dn.maximum;logit=(base/maximum).clamp(1e-6,1-1e-6).logit();origin=logit.sigmoid();limit=oldplan['maximum_logit_shift'];low=(base+maximum*((logit-limit).sigmoid()-origin)).clamp(0,maximum);high=(base+maximum*((logit+limit).sigmoid()-origin)).clamp(0,maximum);positive=dn.orientations>0;negative=~positive;lower=low[:,positive].mean(1)-high[:,negative].mean(1);upper=high[:,positive].mean(1)-low[:,negative].mean(1);locked=(lower>0)|(upper<0);reports={}
  for arm in ['visual_only','visual_native']:
   cp=torch.load(ROOT/f'{arm}-fit-01/ports.pt',map_location='cpu',weights_only=True);s=cp['current_calibration'];cal=VisualDNCurrentCalibration(s['reference'],s['scale'],s['orientations'],rank=4,maximum=maximum,maximum_logit_shift=limit);cal.load_state_dict(s);actual=cal(x,base);response=actual[:,positive].mean(1)-actual[:,negative].mean(1);reports[arm]=dict(actual_current_mean=actual.mean(0).tolist(),actual_opponent_sign_change_fraction=float(((response.sign()!= (base[:,positive].mean(1)-base[:,negative].mean(1)).sign())&locked).float().mean()),minimum_interval_slack=float(torch.minimum(response-lower,upper-response).min()))
  reference_current=float(.2*torch.tanh(torch.tensor(.01)));result=dict(status='completed_analytic_visual_current_range_audit',features_sha256=r['features_sha256'],late_visual_samples=len(x),maximum_logit_shift=limit,original_below_1e_minus5_fraction=(base<1e-5).float().mean(0).tolist(),cannot_cross_zero_opponent_interval_fraction=float(locked.float().mean()),opponent_interval_lower_range=[float(lower.min()),float(lower.max())],opponent_interval_upper_range=[float(upper.min()),float(upper.max())],upper_current_below_reference_command_fraction=(high<reference_current).float().mean(0).tolist(),reference_yaw_increment_rad_s=.01,reference_command_current=reference_current,models=reports,actor_qualified=False,vehicle_control_sent=False,scope='Exact bounds of this bounded logit-correction codec for the observed reserved static features. Proves input-current population ordering cannot reverse in counted samples even with arbitrary correction weights; not a sole-cause, physiological or whole-car controllability proof.')
 write_json(ROOT/'reachable-range-audit-01.json',result);print(result)
if __name__=='__main__':main()
