"""Shared offline experiment assembly; no simulator or vehicle interfaces."""
import copy
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import assemble_student, MuscleVehicleReadout, read
from flybrain_drive.visual_descending_port import VisualDescendingPort
from scripts.trace_sac_premuscle import CHECKPOINT, CHECKPOINT_SHA
from scripts.train_qualified_sac_unified import load_episode

ROOT = Path('runs/visual-circuit-spatial-20261103')
SOURCE = ROOT/'descending-residual-01'
SOURCE_SHA = '26fa0cb994c12f2fe85c3c2b02860e6600634a418b63043ce9f8f43ba5f97618'
DYNAMIC = ('state','traces','previous_drive','previous_scene','previous_raw_scene',
           'previous_time','previous_body_receipt','previous_render_lowers')

def clone(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, tuple): return tuple(clone(x) for x in value)
    return copy.deepcopy(value)

def snapshot(model): return {k:clone(getattr(model,k)) for k in DYNAMIC}
def restore(model, state):
    for k,v in state.items(): setattr(model,k,clone(v))

def dataset():
    entries=read('runs/teacher-synchronized-20261102/student-split-01.json')['episodes']
    items=[load_episode(e['data']) for e in entries]
    result=read(ROOT/'spatial-recovery-01/result.json')
    extras=[]
    for i,(d,c,p) in enumerate(items):
        if p['arrays_sha256'] != entries[i]['arrays_sha256']: raise ValueError('Original split differs')
        path=ROOT/f'spatial-recovery-01/spatial-{i}.npz'
        if sha256(path)!=result['files_sha256'][path.name]: raise ValueError('Spatial source differs')
        with np.load(path,allow_pickle=False) as z:
            joined=z['event_to_camera']
            np.testing.assert_array_equal(z['camera_indices'][joined], d['camera_index'])
            extras.append(np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),
                                          z['discarded_grid8'][joined]],axis=1))
    return entries,items,extras

def assembly(fit=None):
    if sha256(CHECKPOINT)!=CHECKPOINT_SHA or sha256(SOURCE/'spatial-port.pt')!=SOURCE_SHA:
        raise ValueError('Protected original/source checkpoint differs')
    base=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    source=torch.load(SOURCE/'spatial-port.pt',map_location='cpu',weights_only=True)
    if fit is not None:
        root=Path(fit);r=read(root/'result.json')
        if not r['status'].startswith('completed_') or sha256(root/'steering.pt')!=r['checkpoint_sha256']:
            raise ValueError('Exact completed steering fit required')
        source=torch.load(root/'steering.pt',map_location='cpu',weights_only=True)
        if source['source_checkpoint_sha256']!=SOURCE_SHA: raise ValueError('Changed source lineage')
    model,lineage=assemble_student();model.load_state_dict(base['model'],strict=True)
    model.eval().requires_grad_(False);model.motor.backbone.circuit.sparse_backend='cached_csr_cpu'
    _,muscle=model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
    readout=MuscleVehicleReadout(muscle.shape[-1]);readout.load_state_dict(source['readout'],strict=True)
    ac=source['adapter'];adapter=VisualDescendingPort(ac['reference'],ac['scale'],ac['orientations'])
    adapter.load_state_dict(ac,strict=True)
    return model,readout,adapter,base,lineage

def onsets(d):
    """Predeclared body-yaw transition proxy, not track geometry truth."""
    ids=np.flatnonzero(d['fit_mask']); t=d['decision_time']; yaw=d['body'][:,4]
    last=-1e30;out=[]
    for k,j in enumerate(ids):
        if k<25 or k>=len(ids)-10 or t[j]-last<2 or abs(yaw[j])<.15:continue
        prior=ids[(t[ids]>=t[j]-.6)&(t[ids]<t[j])]
        later=ids[(t[ids]>=t[j])&(t[ids]<t[j]+.24)]
        if len(prior)<10 or len(later)<4 or np.mean(abs(yaw[prior]))>=.07 or np.mean(abs(yaw[later]))<=.15:continue
        sign=float(np.sign(np.median(d['labels'][later,0])))
        if not sign:continue
        start=int(np.searchsorted(t,t[j]-1.2))
        out.append(dict(event=int(j),start=start,time=float(t[j]),direction=sign))
        last=t[j]
    return out

def paired_onsets(d):
    events=onsets(d);out=[]
    for x in events:
        candidates=[z for z in events if z['direction']!=x['direction']]
        if not candidates:continue
        # Pair solely for diagnostic stimulus selection; no donor label enters actor.
        donor=min(candidates,key=lambda z:abs(d['body'][z['start'],16]-d['body'][x['start'],16]))
        out.append(dict(**x,donor_event=donor['event'],donor_time=donor['time']))
    return out
