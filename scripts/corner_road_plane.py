"""Estimate a local road plane from road-interior image feature correspondences."""

import cv2
import numpy as np


def fit_plane(cloud, seed=0):
    keep=np.isfinite(cloud).all(axis=1)
    keep&=(cloud[:,0]>=6.)&(cloud[:,0]<=24.)&(abs(cloud[:,1])<=4.)&(abs(cloud[:,2])<=1.5)
    x,y,z=cloud[keep].T
    if len(z)<12:
        return None
    design=np.column_stack((np.ones(len(z)),x,y))
    rng=np.random.default_rng(seed)
    best=None
    for _ in range(32):
        ids=rng.choice(len(z),3,replace=False)
        if abs(np.linalg.det(design[ids]))<.8:
            continue
        co=np.linalg.solve(design[ids],z[ids])
        if abs(co[0])>.8 or np.max(abs(co[1:]))>.15:
            continue
        error=abs(design@co-z)
        inliers=error<=.12
        rank=(int(inliers.sum()),-float(np.median(error)))
        if best is None or rank>best[0]:
            best=rank,inliers
    if best is None:
        return None
    keep=best[1]
    if keep.sum()<12 or keep.mean()<.6 or np.ptp(x[keep])<5. or np.ptp(y[keep])<1.5:
        return None
    result=np.linalg.lstsq(design[keep],z[keep],rcond=None)[0]
    if abs(result[0])>.8 or np.max(abs(result[1:]))>.15:
        return None
    return result,int(keep.sum())


def estimate_plane(previous, current, mount, motion, track_pixels, reconstruct):
    """Use only captured front pixels, fixed calibration and relative vehicle motion."""
    if motion is None or np.linalg.norm(motion[1])<.5:
        return None
    h,w=current.shape
    mask=np.zeros_like(current)
    mask[int(.60*h):int(.94*h),int(.34*w):int(.66*w)]=255
    corners=cv2.goodFeaturesToTrack(current,80,.008,4,mask=mask,blockSize=5)
    if corners is None or len(corners)<12:
        return None
    uv=corners[:,0]/[w-1,h-1]
    old,valid=track_pixels(current,previous,uv)
    if valid.sum()<12:
        return None
    cloud=reconstruct(old[valid],uv[valid],mount,motion)
    return fit_plane(cloud)


def sampling(mount, stations, lateral, plane):
    """Project an estimated road plane without using absolute vehicle pose."""
    x,y=np.meshgrid(stations,lateral,indexing="ij")
    z=plane[0]+plane[1]*x+plane[2]*y
    points=np.stack((x,y,z),axis=-1)
    delta=points-mount["origin_body_m"]
    depth=delta@mount["forward_body"]
    safe=np.where(depth>0.,depth,1.)
    u=.5+.5*mount["projection_x"]*(delta@mount["right_body"])/safe
    v=.5-.5*mount["projection_y"]*(delta@mount["up_body"])/safe
    return u,v,(depth>0.)&(u>=.02)&(u<=.98)&(v>=.02)&(v<=.98)
