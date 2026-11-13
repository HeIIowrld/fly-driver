"""Describe signed body rotation and the shape immediately before and after each candidate."""
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
PHASE = OUT.parent/"vision-corner-phase-confirmation-20261112-01"
sys.path.insert(0,str(PHASE))
from phase_features import PARENT, DEVELOP, SILVER, VAL, group, old, adjacent

HISTORY = 3


def signed_curve(stations, edges, keep):
    x, y = stations[keep],edges[keep]
    if len(x)<3 or x[-1]-x[0]<4. or np.max(np.diff(x))>4.:
        return len(x),0.,0.,0.
    centered = x-x.mean()
    design = np.column_stack((np.ones(len(x)),centered,centered**2))
    co = np.linalg.lstsq(design,y,rcond=None)[0]
    k = 2.*co[2]/(1.+co[1]**2)**1.5
    return len(x),float(k.mean()),float(k[0]-k[1]),float(co[1].mean())


def hinge(x,edges):
    """Compare ongoing constant curvature with a continuous curvature step."""
    if len(x)<6 or np.sum(x<=-1.)<2 or np.sum(x>=1.)<3 or min(x)>-4. or max(x)<4.:
        return (0.,0.,0.,0.,0.)
    smooth = np.column_stack((np.ones(len(x)),x,x**2))
    bend = np.maximum(x,0.)**2
    design = np.column_stack((smooth,bend))
    if np.linalg.cond(design)>1.e4:
        return (0.,0.,0.,0.,0.)
    baseline = np.linalg.lstsq(smooth,edges,rcond=None)[0]
    candidate = np.linalg.lstsq(design,edges,rcond=None)[0]
    error = float(np.sqrt(np.mean((smooth@baseline-edges)**2)))
    improved = float(np.sqrt(np.mean((design@candidate-edges)**2)))
    factor = 2./(1.+candidate[1]**2)**1.5
    change = candidate[3]*factor
    return (1.,float(change.mean()),float(change[0]-change[1]),error,error-improved)


def relative_features(stations,bounds,motion,prior,offer):
    direction,distance = offer
    orient = 1. if direction==1 else -1.
    edges = bounds if orient>0 else -bounds[:,::-1]
    valid = np.isfinite(edges).all(1)
    width = edges[:,1]-edges[:,0]
    valid &= (width>3.)&(width<20.)
    _,speed,yaw = motion
    signed_yaw = orient*yaw
    body = signed_yaw/max(speed,1.)
    old_body = orient*prior[:,2]/np.maximum(prior[:,1],1.)
    near = signed_curve(stations,edges,valid&(stations<=10.))
    x = stations-distance
    before = signed_curve(stations,edges,valid&(x>=-12.)&(x<=0.))
    after = signed_curve(stations,edges,valid&(x>=0.)&(x<=12.))
    use = valid&(abs(x)<=12.)
    fit = hinge(x[use],edges[use])
    return np.asarray([signed_yaw,body,len(prior)-1,float(old_body.mean()),body-float(old_body[0]),
                       *near,*before,*after,*fit,after[1]-before[1],
                       before[1]-body,after[1]-body,near[1]-body],dtype=float)


def causal_prefix(rows,motion,i):
    start = i
    while (i-start<HISTORY and start>0 and adjacent(rows[start-1],rows[start])
           and np.isfinite(motion[start-1]).all() and motion[start-1,1]>=old.MIN_SPEED_MPS):
        start -= 1
    return motion[start:i+1]


def enrich(track,sources):
    g = group(track,sources)
    path = PARENT/f"{track}-observations.npz"
    proof = old.read(path.with_suffix(".json"))
    assert old.digest(path)==proof["arrays_sha256"]
    with np.load(path,allow_pickle=False) as data:
        assert list(zip(data["run"],data["frame"],strict=True))==g["keys"]
        motion = data["motion"]
    assert np.isfinite(motion[g["index"]]).all()
    directional = np.stack([relative_features(old.STATIONS,g["sections"][i],motion[i],
                             causal_prefix(g["rows"],motion,i),offer)
                            for i,offer in zip(g["index"],g["offers"],strict=True)])
    x = g["phase_x"][g["index"]]
    assert np.isfinite(x).all() and np.isfinite(directional).all()
    sources.update((Path(__file__),path,path.with_suffix(".json"),Path(adjacent.__code__.co_filename)))
    return dict(g,edge=np.c_[g["edge"],x,directional],directional=directional)
