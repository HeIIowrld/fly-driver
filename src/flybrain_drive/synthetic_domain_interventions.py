"""Exploratory deterministic front-view image perturbations for synthetic diagnostics.

These screen-space overlays are not a 3-D track, physical cockpit, or weather model.
"""
import numpy as np

MODES=('baseline','front_exposure','front_sky','front_roadside','front_cockpit')

def intervene(views, mode):
    if mode not in MODES or not isinstance(views,(list,tuple,np.ndarray)) or len(views)!=4:
        raise ValueError('Four ordered RGB views and known intervention required')
    if any(not isinstance(v,np.ndarray) or v.shape!=(180,320,3) or v.dtype!=np.uint8 for v in views):
        raise ValueError('Four 320x180 uint8 RGB views required')
    out=[v.copy() for v in views];front=out[0]
    if mode=='front_exposure':
        out[0]=np.rint(front.astype(np.float32)*.75).clip(0,255).astype(np.uint8)
    elif mode=='front_sky':
        # Fixed screen-space cloud texture is diagnostic, not physical weather.
        y,x=np.indices((55,320));wave=np.sin(x/19.)+np.cos(y/9.)+np.sin((x+y)/31.)
        cloud=np.clip(170+30*wave,70,245).astype(np.uint8)
        front[:55]=np.stack((cloud,cloud,cloud),axis=-1)
    elif mode=='front_roadside':
        # Symmetric abstract poles; positions are screen fixed, not track geometry.
        for x in (12,28,48,272,292,308):
            front[35:118,x:x+3]=np.array([45,47,42],dtype=np.uint8)
        front[75:78,:75]=np.array([80,80,76],dtype=np.uint8)
        front[75:78,245:]=np.array([80,80,76],dtype=np.uint8)
    elif mode=='front_cockpit':
        # Abstract body-fixed silhouette; no real cockpit pixels are used.
        y,x=np.indices((180,320));mask=(y>=145)|((y>=125)&(np.abs(x-160)<(y-125)*3))
        front[mask]=np.array([35,38,40],dtype=np.uint8)
    return out
