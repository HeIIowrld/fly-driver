"""TRAIN-only normalization and information-matched coarse motion control."""
import numpy as np


def duplicate_coarse_motion(scene):
    scene=np.asarray(scene)
    if scene.ndim!=2 or scene.shape[1]!=576 or not np.isfinite(scene).all():raise ValueError('Finite original four-view576 scene required')
    motion=scene.reshape(-1,4,9,16)[:,:,1:].reshape(-1,4,8,4,4)
    return np.repeat(np.repeat(motion,2,axis=-2),2,axis=-1).reshape(-1,2048).astype(np.float32)


def training_motion_normalization(features,event_maps,fit_masks,*,floor=.005):
    if len(features)!=3 or len(event_maps)!=3 or len(fit_masks)!=3 or not np.isfinite(floor) or floor<=0:raise ValueError('Two TRAIN/one development arrays and explicit positive floor required')
    selected=[]
    for i in (0,1):
        x=np.asarray(features[i]);j=np.asarray(event_maps[i]);mask=np.asarray(fit_masks[i])
        if (x.ndim!=2 or x.shape[1]!=2048 or not np.isfinite(x).all() or j.ndim!=1 or mask.shape!=j.shape
                or j.dtype.kind not in 'iu' or mask.dtype!=np.bool_ or not mask.any() or np.any(j<0) or np.any(j>=len(x))):raise ValueError('Valid TRAIN camera indices and original supervision mask required')
        selected.append(x[j[mask]].astype(float))
    selected=np.concatenate(selected)
    return selected.mean(0).astype(np.float32),np.maximum(selected.std(0),floor).astype(np.float32)
