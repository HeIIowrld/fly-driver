"""Audited offline native/visual assembly shared by matched experiments."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_native_flight_candidate import load_audited_native_flight_candidate
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode
from scripts.fit_bounded_sac_steering import pre_mask


def load_experiment(reference_plan):
    plan=read(reference_plan);transfer=read(plan['transfer_plan'])
    if sha256(plan['transfer_plan'])!=plan['transfer_plan_sha256']:raise ValueError('Frozen transfer plan changed')
    if sha256(transfer['split'])!=transfer['split_sha256']:raise ValueError('Frozen cohort changed')
    components,receipt=load_audited_native_flight_candidate(plan['native'])
    model,head,dn,optical,port,encoder=components
    cfg=read(Path(plan['native']['fit'])/'manifest.json')['config']
    entries=three_episode_cohort(transfer['split'])
    if entries!=cfg['entries']:raise ValueError('Native/visual cohorts differ')
    items=[load_episode(e['data']) for e in entries]
    spatial=Path(transfer['spatial']);sr=read(spatial/'result.json')
    extension=Path(cfg['arguments']['extension']);er=read(extension/'result.json')
    sign=read(Path(transfer['candidate'])/'manifest.json')['config']['sign_evidence']
    if not sign['training_only'] or sign['agreement']<.9:raise ValueError('TRAIN sign evidence missing')
    extras=[];native=[];valid=[];pre=[]
    for e,(d,_,evidence) in enumerate(items):
        if evidence['arrays_sha256']!=entries[e]['arrays_sha256']:raise ValueError('Actor arrays changed')
        sp=spatial/f'spatial-{e}.npz';ep=extension/f'extension-{e}.npz'
        if sha256(sp)!=sr['files_sha256'][sp.name] or sha256(ep)!=er['files_sha256'][ep.name]:raise ValueError('Sidecar changed')
        with np.load(sp,allow_pickle=False) as z:
            j=z['event_to_camera'];np.testing.assert_array_equal(z['camera_indices'][j],d['camera_index'])
            extras.append(torch.tensor(np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),z['discarded_grid8'][j]],1),dtype=torch.float32))
        with np.load(ep,allow_pickle=False) as z:
            for key in ('body_packet','body_receipt','decision_time'):np.testing.assert_array_equal(z[key],d[key])
            native.append(torch.tensor(z['values']));valid.append(torch.tensor(z['valid']))
        pre.append(pre_mask(d,sign['sign']))
    if sha256(Path(plan['reference'])/'predictions.npz')!=plan['reference_predictions_sha256']:raise ValueError('Reference trajectory changed')
    with np.load(Path(plan['reference'])/'predictions.npz',allow_pickle=False) as z:
        baseline=[z[f'candidate_{e}'].copy() for e in range(3)]
        muscles=[z[f'muscles_{e}'].copy() for e in range(3)]
        for e in range(3):np.testing.assert_array_equal(pre[e],z[f'pre_{e}'])
    return components,receipt,entries,items,extras,native,valid,pre,baseline,muscles


def tensor_state_equal(left,right):
    if set(left)!=set(right):return False
    for k,v in left.items():
        other=right[k]
        if v.layout!=other.layout or v.shape!=other.shape or v.dtype!=other.dtype:return False
        if v.layout!=torch.strided:
            a,b=v.to_sparse_coo().coalesce(),other.to_sparse_coo().coalesce()
            if not torch.equal(a.indices(),b.indices()) or not torch.equal(a.values(),b.values()):return False
        elif not torch.equal(v,other):return False
    return True
