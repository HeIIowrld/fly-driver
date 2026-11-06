"""Outer current intervals at learned gates, distinct from motor controllability."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.visual_descending_port import VisualDescendingPort
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from flybrain_drive.full_range_port_candidate import FullRangeCurrent

ROOT=Path('runs/full-range-port-learning-20261104')

def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');original=torch.load('runs/motion-spatial-learning-20261104/fine_motion-fit-01/motion-spatial.pt',map_location='cpu',weights_only=True)['adapter'];dn=VisualDescendingPort(original['reference'],original['scale'],original['orientations']);dn.load_state_dict(original);source=ROOT/'reserved-stimuli-01';r=read(source/'result.json');assert sha256(source/'features.npz')==r['features_sha256'];models={}
 with np.load(source/'features.npz') as z:x=torch.tensor(z['static'][:,25:].reshape(-1,320))
 with torch.no_grad():
  parent=dn(x);positive=dn.orientations>0;negative=~positive
  for arm in ['visual_only','visual_native']:
   fit=ROOT/f'{arm}-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);s=cp['current_calibration'];base=VisualDNCurrentCalibration(s['base.reference'],s['base.scale'],s['base.orientations'],rank=4,maximum=dn.maximum);current=FullRangeCurrent(base,p['initial_gate']);current.load_state_dict(s);pilot=current.base(x,parent);g=current.registration.gate;actual=current(x,parent);low=(1-g)*pilot;high=low+g*dn.maximum;lower=low[:,positive].mean(1)-high[:,negative].mean(1);upper=high[:,positive].mean(1)-low[:,negative].mean(1);opponent=actual[:,positive].mean(1)-actual[:,negative].mean(1);assert (actual>=low-1e-7).all() and (actual<=high+1e-7).all()
   models[arm]=dict(gates=g.tolist(),actual_current_mean=actual.mean(0).tolist(),actual_opponent_range=[float(opponent.min()),float(opponent.max())],fixed_gate_outer_interval_excludes_zero_fraction=float(((lower>0)|(upper<0)).float().mean()),outer_lower_range=[float(lower.min()),float(lower.max())],outer_upper_range=[float(upper.min()),float(upper.max())])
 write_json(ROOT/'current-range-audit-01.json',dict(status='completed_learned_gate_current_outer_interval_audit',visual_samples=len(x),features_sha256=r['features_sha256'],models=models,scope='Outer bounds allow independent target currents anywhere in [0,.2] while holding learned gates and frozen pilot fixed. Exclusion proves no ordering reversal at those gates; inclusion does not prove weight reachability, vehicle control, or sole failure cause. Gate itself can be retrained.',actor_qualified=False,vehicle_control_sent=False));print(models)
if __name__=='__main__':main()
