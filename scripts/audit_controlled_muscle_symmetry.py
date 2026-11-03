"""Evaluate frozen synthetic symmetry probes on pre-existing controlled stimuli."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs'); d=B/'20261101-muscle-symmetry-diagnostic-1'; c=B/'20261101-controlled-flight-probe-1'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
dr=read(d/'result.json');cr=read(c/'result.json');dp=d/'diagnostic.npz';cp=c/'replication.npz'
if dr['status']!='completed_synthetic_muscle_symmetry_diagnostic' or cr['status']!='completed_synthetic_frozen_probe_replication' or sha256(dp)!=dr['output_sha256'] or sha256(cp)!=cr['output_sha256']:raise ValueError('exact evidence required')
run=begin_run(B/'20261101-muscle-symmetry-controlled-audit-1',dict(program_sha256=sha256(__file__),diagnostic_sha256=sha256(dp),controlled_sha256=sha256(cp),probe_refits=0))
r=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,probe_refits=0)
try:
 with np.load(dp,allow_pickle=False) as z:center=z['center'].copy();weights={k:z[k+'_weights'].copy() for k in ('linear','absolute_bilateral')}
 def score(x,model):
  z=(x-center)/1e-6
  if model=='absolute_bilateral':z=np.concatenate([z,np.abs(z[:,[1,3,5]]-z[:,[0,2,4]])],axis=1)
  return np.column_stack((z,np.ones(len(z))))@weights[model]
 results={}
 with np.load(cp,allow_pickle=False) as data:
  for item in cr['conditions']:
   label=item['condition']['comparison'].replace(' ','_');x=data[label+'_muscles6_intact'].astype('float64');lesion=data[label+'_muscles6_lesion'].astype('float64');models={}
   for model in weights:
    pred=score(x,model);off=score(lesion,model);means=pred.reshape(3,10).mean(axis=1);offmeans=off.reshape(3,10).mean(axis=1)
    perturb=[]
    for axis in range(6):
     noise=np.zeros_like(x);noise[:,axis]=1e-6
     for sign in (-1,1):perturb.append(float(np.max(np.abs(score(x+sign*noise,model)-pred))))
    models[model]=dict(means=means.tolist(),straight_minus_bend=float(means[0]-means[1:].mean()),spatial_off_means=offmeans.tolist(),max_output_change_on_one_coordinate_shift_1e_6=max(perturb))
   results[label]=dict(condition=item['condition'],models=models)
 r.update(status='completed_frozen_symmetry_controlled_audit',conditions=results,diagnostic_sha256=sha256(dp),controlled_sha256=sha256(cp),interpretation='Post-hoc correlated synthetic states, deterministic 1e-6 input shifts; no retraining, no sensor noise model or causal closed-loop claim')
except Exception as e:r.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',r)
print(json.dumps(r,allow_nan=False))
if r['status']!='completed_frozen_symmetry_controlled_audit':raise SystemExit(1)
