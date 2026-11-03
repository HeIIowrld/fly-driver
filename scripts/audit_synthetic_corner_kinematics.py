"""Describe stored synthetic closed-loop steering relative to centerline geometry.

Centerline curvature and pose are offline diagnostic truth, never actor inputs.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
root=Path('D:/CodexResearchRuns/flybrain-synthetic-load-loop-20261102-01');result=json.loads((root/'result.json').read_text());data=root/'trajectory.npz';assert hashlib.sha256(data.read_bytes()).hexdigest()==result['output_sha256'] and result['status']=='completed_synthetic_load_closed_loop'
rows=[]
with np.load(data,allow_pickle=False) as z:
 for speed in (3,6):
  for bend in (-1,1):
   a=z[f'{speed}_{bend}_base'];assert np.all(np.diff(a[:,1])>0)
   for x in (13.,14.,15.,15.8):
    if not a[0,1]<=x<=a[-1,1]:raise ValueError('insufficient original trajectory')
    slope=bend*.036*max(x-12.,0.);curvature=bend*.036/(1+slope*slope)**1.5;ideal_steer=math.atan(2.5*curvature);steer=float(np.interp(x,a[:,1],a[:,7]));yaw=float(np.interp(x,a[:,1],a[:,3]));ideal_yaw=math.atan(slope);center=bend*.018*max(x-12.,0.)**2;y=float(np.interp(x,a[:,1],a[:,2]));rows.append({'speed':speed,'bend':bend,'x_m':x,'actual_steer_rad':steer,'centerline_feedforward_steer_rad':ideal_steer,'actual_yaw_rad':yaw,'centerline_tangent_yaw_rad':ideal_yaw,'lateral_error_m':y-center})
out=Path('D:/CodexResearchRuns/flybrain-corner-kinematic-audit-20261102-01');out.mkdir(exist_ok=False);report={'status':'completed_posthoc_synthetic_kinematic_audit','source_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'interpretation':'Ideal centerline steering assumes the vehicle is exactly on the tangent and is not a corrective steering target. World geometry and state are offline diagnostic only, never actor inputs. Slip-free synthetic bicycle, no AC or physiological claim.'};(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8')
for r in rows:
 if r['x_m']==15.8:print(r)
