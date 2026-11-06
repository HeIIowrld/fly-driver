"""Audited fine-motion data and same-information coarse control at fixed body state."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.motion_spatial_residual_port import MOTION_CELL_TYPES
from flybrain_drive.motion_spatial_training_data import duplicate_coarse_motion,training_motion_normalization
from flybrain_drive.sac_unified_student import read
from scripts.new_condition_joint_experiment import load_joint_experiment


def load_motion_experiment(plan,arm):
    if arm not in ('fine_motion','coarse_duplicate'):raise ValueError('Declared matched motion arm required')
    for path,expected in plan['source_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Sealed source changed '+path)
    if sha256(plan['reference_plan'])!=plan['reference_plan_sha256']:raise ValueError('Frozen parent experiment changed')
    components,receipt,entries,items,extras,native,valid,baseline,pre=load_joint_experiment(read(plan['reference_plan']))
    features=[];maps=[];coarse_refs=[]
    for e,(entry,(d,c,_),spec) in enumerate(zip(entries,items,plan['motion'])):
        root=Path(spec['recovery']);r=read(root/'result.json');m=read(root/'manifest.json')['config'];audit=Path(spec['audit']);ar=read(audit/'result.json');am=read(audit/'manifest.json')['config']
        if (r['status']!='completed_qualified_motion_spatial_recovery' or ar['status']!='completed_independent_qualified_motion_spatial_audit'
                or am['recovery_result_sha256']!=sha256(root/'result.json') or m['actor_arrays_sha256']!=entry['arrays_sha256']
                or sha256(root/'motion-spatial.npz')!=r['features_sha256'] or tuple(r['modeled_cell_types'])!=MOTION_CELL_TYPES
                or list(r['view_order'])!=list(VIEW_ORDER)):raise ValueError('Exact qualified direction/view motion provenance required')
        with np.load(root/'motion-spatial.npz') as z:
            ids=z['camera_indices'].copy();j=z['event_to_camera'].copy();np.testing.assert_array_equal(ids[j],d['camera_index']);coarse=z['original_motion_pool'].copy();fine=z['discarded_motion_grid8'].copy()
        first=np.array([np.flatnonzero(d['camera_index']==i)[0] for i in ids]);original_coarse=d['scene'][first].reshape(-1,4,9,16)[:,:,1:].reshape(-1,512)
        np.testing.assert_allclose(coarse,original_coarse,atol=2e-6,rtol=0)
        feature=fine if arm=='fine_motion' else duplicate_coarse_motion(d['scene'][first])
        if feature.shape!=(len(ids),2048) or not np.isfinite(feature).all():raise ValueError('Finite complete motion data required')
        features.append(feature);maps.append(j);coarse_refs.append(coarse)
    reference,scale=training_motion_normalization(features,maps,[d['fit_mask'] for d,_,_ in items],floor=plan['normalization_floor'])
    return (components,receipt,entries,items,extras,native,valid,baseline,pre,
            [torch.tensor(x,dtype=torch.float32) for x in features],maps,coarse_refs,torch.tensor(reference),torch.tensor(scale))
