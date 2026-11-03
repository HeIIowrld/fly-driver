"""Audit matched synthetic roll/yaw signal magnitudes through saved flight states."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
rows=[]
for seed in (882001,882002):
 for axis in ('roll','yaw'):
  scene=B/(f'20261101-synthetic-roll-road-{seed}' if axis=='roll' else f'20261102-synthetic-yaw005-road-{seed}')
  state=B/(f'20261102-roll-populations-{seed}-fixed' if axis=='roll' else f'20261102-yaw005-{seed}-lesion-none')
  sr=read(scene/'result.json');tr=read(state/'result.json');sf=scene/'responses.npz';tf=state/'outputs.npz'
  if sha256(sf)!=sr['responses_sha256'] or sha256(tf)!=tr['output_sha256'] or tr.get('vehicle_control_sent') is not False:raise ValueError('Exact saved sources required')
  with np.load(sf,allow_pickle=False) as z:
   retina=z['retina'][0];motion=z['motion'][0];dret=retina[2,-10:]-retina[1,-10:];dmot=motion[2,-10:]-motion[1,-10:]
   retinal=np.sqrt(np.mean(dret**2,axis=(0,2)));moving=np.sqrt(np.mean(dmot**2,axis=(0,2,3)))
  with np.load(tf,allow_pickle=False) as z:
   groups={}
   for name in ('HS','VS','DNg02','wing_outputs','muscles'):
    delta=z[name][2,-10:]-z[name][1,-10:];groups[name]=dict(rms=float(np.sqrt(np.mean(delta**2))),mean_signed=float(np.mean(delta)))
   delta=z['outputs'][0,2,-10:,0]-z['outputs'][0,1,-10:,0]
  rows.append(dict(seed=seed,axis=axis,source_sha256=sha256(sf),state_sha256=sha256(tf),retinal_rms_by_view=retinal.tolist(),motion_rms_by_view=moving.tolist(),groups=groups,turn_mean_last_10=float(delta.mean()),turn_rms_last_10=float(np.sqrt(np.mean(delta**2)))))
run=begin_run(B/'20261102-synthetic-roll-yaw-signal-audit-1',dict(program_sha256=sha256(__file__),frames='last 10/41',actor=False,optimizer_updates=0));out=dict(status='completed_synthetic_roll_yaw_signal_audit',conditions=rows,actor_qualified=False,vehicle_control_sent=False,interpretation='Two textures, one synthetic planar scene, same nominal angular rate and fixed body; input geometry differs and group RMS uses different units; no physiology or driving')
write_json(run/'result.json',out);print(json.dumps(out))
