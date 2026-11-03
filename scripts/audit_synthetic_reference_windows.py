"""Offline paired reference-window sensitivity on retained synthetic FlyVis responses."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
b=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
paths=[b/'20261101-flyvis-coherent-four-view-yaw',*(b/f'20261101-flyvis-coherent-yaw-texture-{n}' for n in range(863102,863108))]
camera=Path('configs/ac-csp-exterior-front-v3-retina.json');registration=b/'20261031-pace-train-flyvis-replay/retina-registration.npz'
with np.load(registration,allow_pickle=False) as f:
 pool=VisualSelfMotionPool.from_registration(json.loads(camera.read_text(encoding='utf8')),{k:f[k].copy() for k in f.files})
run=begin_run(a.output,dict(program_sha256=sha256(__file__),camera_sha256=sha256(camera),registration_sha256=sha256(registration),
 sources=[dict(path=str(x),sha256=sha256(x/'responses.npz')) for x in paths],
 comparison='same existing first 11 stationary frames; full mean versus last three mean; no fitting',optimizer_updates=0))
rows=[]
for path in paths:
 r=json.loads((path/'result.json').read_text(encoding='utf8'))
 if r['status']!='completed_coherent_four_view_yaw' or r['responses_sha256']!=sha256(path/'responses.npz'):raise ValueError('Synthetic source changed')
 with np.load(path/'responses.npz') as f:
  motion=f['motion'].copy(); t=f['times'].copy(); yaw=f['yaw_rad_s'].copy()
 if motion.shape[:2]!=(3,41) or not np.array_equal(yaw,np.array([0.,-.23,.23])):raise ValueError('Original synthetic cases required')
 features=pool.features(motion)[:,:,2]
 metrics={}
 for label,ref in [('mean11',features[:,:11].mean(axis=1)),('last3',features[:,8:11].mean(axis=1))]:
  response=features[:,11:]-ref[:,None]
  final=response[:,-10:].mean(axis=1);relative=final[1:]-final[0]
  metrics[label]=dict(stationary_mean_last10=float(final[0]), rotation_mean_last10=final[1:].tolist(),relative_to_stationary=relative.tolist(),opposite_sign=bool(relative[0]*relative[1]<0))
 rows.append(dict(path=str(path),cases=yaw.tolist(),metrics=metrics))
write_json(run/'result.json',dict(status='completed_synthetic_reference_window_comparison',rows=rows,
 physical_exposure_or_physiology_verified=False,optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False))
print(json.dumps(rows))
