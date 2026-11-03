"""Perturb a frozen multi-speed synthetic probe; no new fit or vehicle control."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
b=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');src=b/'20261101-multispeed-symmetry-pilot-1';r=json.loads((src/'result.json').read_text());p=src/'multispeed-pilot.npz'
if r['status']!='completed_synthetic_multispeed_symmetry_pilot' or sha256(p)!=r['output_sha256']:raise ValueError('exact trained diagnostic required')
run=begin_run(b/'20261101-multispeed-symmetry-perturbation-1',dict(program_sha256=sha256(__file__),source_sha256=sha256(p),probe_refits=0));out=dict(status='failed',vehicle_control_sent=False,actor_qualified=False)
try:
 with np.load(p,allow_pickle=False) as z:
  c=z['center_coeff'].copy();w=z['weights'].copy();xs={speed:z[f'{speed:g}_evaluation_muscles'].astype('float64') for speed in (1.5,3.,4.5)}
 def predict(x,speed):
  z=(x-(np.array([1.,speed])@c))/1e-6;d=np.column_stack((z,np.abs(z[:,[1,3,5]]-z[:,[0,2,4]]),np.ones(len(z))));return np.clip(d@w,0.,5.)
 rows=[]
 for speed,x in xs.items():
  base=predict(x,speed);muscle=[]
  for i in range(6):
   delta=np.zeros_like(x);delta[:,i]=1e-6
   for sign in (-1,1):muscle.append(float(np.max(np.abs(predict(x+sign*delta,speed)-base))))
  speed_shifts=[float(np.max(np.abs(predict(x,speed+sign*.1)-base))) for sign in (-1,1)]
  rows.append(dict(speed_m_s=speed,max_change_muscle_1e_6=max(muscle),max_change_speed_input_0_1=max(speed_shifts)))
 out.update(status='completed_synthetic_multispeed_perturbation',conditions=rows,probe_refits=0,interpretation='Counterfactual fixed feature perturbations and +/-0.1m/s speed input mismatch; no measured noise or physical closed-loop')
except Exception as e:out.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',out)
print(json.dumps(out))
if out['status']!='completed_synthetic_multispeed_perturbation':raise SystemExit(1)
