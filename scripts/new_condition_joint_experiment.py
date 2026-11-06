"""Qualified new-condition sensory data; frozen audited baseline, no actor truth."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_native_visual_candidate import load_readonly_native_candidate
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode
from scripts.fit_bounded_sac_steering import pre_mask


def load_joint_experiment(plan):
    for name,expected in plan['source_sha256'].items():
        if sha256(name)!=expected:raise ValueError('Sealed source changed '+name)
    for name,key in [('split','split_sha256'),('candidate_plan','candidate_plan_sha256')]:
        if sha256(plan[name])!=plan[key]:raise ValueError('Sealed data/candidate changed')
    entries=three_episode_cohort(plan['split']);items=[load_episode(e['data']) for e in entries];components,receipt=load_readonly_native_candidate(read(plan['candidate_plan']))
    candidate=read(plan['candidate_plan']);old_cfg=read(Path(candidate['native']['fit'])/'manifest.json')['config'];old_entries=three_episode_cohort(old_cfg['arguments']['split'])
    if entries[0]!=old_entries[0] or entries[2]!=old_entries[2]:raise ValueError('Original cold/development roles must remain exact')
    reference=Path(candidate['visual']['fit']);rr=read(reference/'result.json')
    if sha256(reference/'predictions.npz')!=rr['predictions_sha256']:raise ValueError('Audited original visual trajectories changed')
    evaluation=Path(plan['new_evaluation']);er=read(evaluation/'result.json');em=read(evaluation/'manifest.json')['config'];audit=Path(plan['new_evaluation_audit']);ar=read(audit/'result.json');am=read(audit/'manifest.json')['config']
    if er['status']!='completed_qualified_native_candidate_evaluation' or er['case']!='normal' or em['entry']!=entries[1] or em['candidate']!=receipt or ar['status']!='completed_independent_qualified_native_candidate_audit' or am['evaluation_result_sha256']!=sha256(evaluation/'result.json') or sha256(evaluation/'predictions.npz')!=er['predictions_sha256']:raise ValueError('Independently audited new baseline required')
    spatial=Path(plan['spatial']);sr=read(spatial/'result.json');sm=read(spatial/'manifest.json')['config']
    if sm['entries']!=entries or sr['status']!='completed_qualified_static_spatial_recovery':raise ValueError('Same-cohort spatial evidence required')
    extension=Path(old_cfg['arguments']['extension']);xr=read(extension/'result.json');extras=[];values=[];valid=[];baseline=[];pre=[]
    with np.load(reference/'predictions.npz') as z:old_baseline={e:z[f'prediction_{e}'].copy() for e in (0,2)}
    for e,(d,c,provenance) in enumerate(items):
        if provenance['arrays_sha256']!=entries[e]['arrays_sha256']:raise ValueError('Actor arrays changed')
        sp=spatial/f'spatial-{e}.npz';assert sha256(sp)==sr['files_sha256'][sp.name]
        with np.load(sp) as z:
            j=z['event_to_camera'];np.testing.assert_array_equal(z['camera_indices'][j],d['camera_index']);extras.append(torch.tensor(np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),z['discarded_grid8'][j]],1),dtype=torch.float32))
        if e==1:
            with np.load(evaluation/'predictions.npz') as z:values.append(torch.tensor(z['native_values']));valid.append(torch.tensor(z['native_valid']));baseline.append(z['prediction'].copy())
        else:
            path=extension/f'extension-{e}.npz';assert sha256(path)==xr['files_sha256'][path.name]
            with np.load(path) as z:
                for key in ['body_packet','body_receipt','decision_time']:np.testing.assert_array_equal(z[key],d[key])
                values.append(torch.tensor(z['values']));valid.append(torch.tensor(z['valid']))
            baseline.append(old_baseline[e])
        pre.append(pre_mask(d,plan['steering_sign']))
        if len(baseline[e])!=len(d['labels']) or not pre[-1].any():raise ValueError('Complete baseline and explicit pre-turn labels required')
    return components,receipt,entries,items,extras,values,valid,baseline,pre
