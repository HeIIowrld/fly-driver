"""Diagnose fixed geometry-weighted FlyVis flow across preassigned synthetic textures."""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--fresh-probe',action='store_true',help='Also evaluate predeclared fresh seed 863106')
    a=p.parse_args()
    base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    camera_file=Path('configs/ac-csp-exterior-front-v3-retina.json')
    camera=read(camera_file)
    registration=base/'20261031-pace-train-flyvis-replay/retina-registration.npz'
    sources=[(863101,'train',base/'20261101-flyvis-coherent-four-view-yaw'),
             (863103,'train',base/'20261101-flyvis-coherent-yaw-texture-863103'),
             (863104,'train',base/'20261101-flyvis-coherent-yaw-texture-863104'),
             (863102,'previous_known_dev',base/'20261101-flyvis-coherent-yaw-texture-863102'),
             (863105,'known_dev',base/'20261101-flyvis-coherent-yaw-texture-863105')]
    if a.fresh_probe:
        sources.append((863106,'fresh_probe',base/'20261101-flyvis-coherent-yaw-texture-863106'))
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),camera_sha256=sha256(camera_file),
        registration_sha256=sha256(registration),sources=[dict(seed=s,role=r,
        responses_sha256=sha256(p/'responses.npz')) for s,r,p in sources],
        scope='fixed geometry response diagnostic; no calibration, fitting, or vehicle control'))
    with np.load(registration,allow_pickle=False) as f:
        pool=VisualSelfMotionPool.from_registration(camera,{k:f[k].copy() for k in f.files})
    values=[]; reports=[]
    for seed,role,path in sources:
        meta,proof=read(path/'manifest.json')['config'],read(path/'result.json')
        if (meta['texture_seed'] != seed or proof['status'] != 'completed_coherent_four_view_yaw'
                or sha256(path/'responses.npz') != proof['responses_sha256']
                or proof['cases'] != [0.,-.23,.23]):
            raise ValueError('Role-pinned synthetic source mismatch')
        with np.load(path/'responses.npz',allow_pickle=False) as f:
            activity=f['motion'].copy();times=f['times'].copy()
        features=pool.features(activity)
        if features.shape != (3,41,6):
            raise ValueError('Original continuous four-view responses required')
        yaw=features[:,-10:,2].mean(1)
        delta=yaw[1:]-yaw[0]
        reports.append(dict(seed=seed,role=role,yaw_feature_mean=yaw.tolist(),
            relative_to_static=delta.tolist(),opposite_direction_separation=float(delta[1]-delta[0]),
            opposite_signs=bool(delta[0]*delta[1]<0)))
        values.append(features)
    np.savez_compressed(run/'features.npz',features=np.stack(values),times=times)
    write_json(run/'result.json',dict(status='completed_fixed_geometric_motion_diagnostic',
        reports=reports,features_sha256=sha256(run/'features.npz'),
        calibration_or_optimizer_updates=0,actor_deployed=False,
        warning='Geometry weights are engineering hypotheses, not measured fly synapses; '
                'known DEV used descriptively, not final validation.'))
    print(json.dumps(reports))


if __name__=='__main__':
    main()
