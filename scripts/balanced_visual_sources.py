"""Explicitly promoted consumed synthetic sources; original evidence unchanged."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def load_balanced_sources(plan,parent):
    sources=[]
    for spec in plan['synthetic_sources']:
        root=Path(spec['stimuli']);baseline=Path(spec['response']);r=read(root/'result.json');m=read(root/'manifest.json')['config'];br=read(baseline/'result.json');bm=read(baseline/'manifest.json')['config']
        if spec['current_role']!='train_this_trial' or sha256(root/'result.json')!=spec['stimuli_result_sha256'] or sha256(baseline/'result.json')!=spec['response_result_sha256'] or r['status']!='completed_controlled_motion_factorial' or br['status']!='completed_controlled_motion_factorial_response' or sha256(root/'features.npz')!=r['features_sha256'] or sha256(baseline/'responses.npz')!=br['responses_sha256'] or bm['candidate']!=parent or bm['stimuli_result_sha256']!=sha256(root/'result.json'):raise ValueError('Exact explicitly promoted TRAIN source/parent anchors required')
        with np.load(root/'features.npz') as z:x={k:z[k].copy() for k in z.files}
        with np.load(baseline/'responses.npz') as z:anchor={k:z[k].copy() for k in z.files if k.endswith('_fine_prediction')}
        if len(m['cases'])!=12 or m['plan']['temperature_c']!=[40.,80.,110.] or list(x['case_bend'])!=[-1,1]*6:raise ValueError('Matched twelve-case/three-temperature design required')
        sources.append(dict(x=x,metadata=m,plan=m['plan'],anchor=anchor))
    if len(sources)!=3:raise ValueError('Three declared TRAIN nuisance groups required')
    return sources
