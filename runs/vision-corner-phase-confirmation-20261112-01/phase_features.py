"""Frame-local phase features from image boundaries and causal body sensing."""
from pathlib import Path
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent/"vision-corner-temporal-alternatives-20261112-01"
sys.path.insert(0,str(PARENT))
from alternatives import DEVELOP, SILVER, VAL, data, old, adjacent

CLASS = {"straight":0,"out_of_range":0,"approach":1,"in_corner":2}
HISTORY = 3


def near_shape(sections,low,high):
    station = old.STATIONS
    width = sections[:,1]-sections[:,0]
    valid = (station>=low)&(station<=high)&np.isfinite(sections).all(1)&(width>3.)&(width<20.)
    x, values = station[valid], sections[valid]
    n = len(x)
    if n<3:
        return [float(n),0.,0.,0.,0.,0.,0.]
    centered = x-x.mean()
    design = np.column_stack((np.ones(n),centered,centered**2))
    coef = np.linalg.lstsq(design,values,rcond=None)[0]
    curvature = 2.*coef[2]/(1.+coef[1]**2)**1.5
    return [float(n), abs(float(coef[1].mean())), abs(float(curvature.mean())),
            abs(float(curvature[1]-curvature[0])), float(width[valid].mean()),
            float(width[valid].std()), float(np.sqrt(np.mean((design@coef-values)**2)))]


def current_features(bounds,motion):
    speed,yaw = motion[1:]
    rate = yaw/max(speed,1.)
    return np.asarray([speed,abs(yaw),abs(rate),*near_shape(bounds,2.,8.),
                       *near_shape(bounds,4.,14.),*near_shape(bounds,10.,24.)])


def frame_features(rows,sections,motion):
    moving = np.isfinite(motion).all(1)&(motion[:,1]>=old.MIN_SPEED_MPS)
    values = np.full((len(rows),3+3*7),np.nan)
    for i in np.flatnonzero(moving):
        values[i] = current_features(sections[i],motion[i])
    output = np.full((len(rows),values.shape[1]+8),np.nan)
    for i in np.flatnonzero(moving):
        start = i
        while (i-start < HISTORY and start>0 and moving[start-1]
               and adjacent(rows[start-1],rows[start])):
            start -= 1
        past = values[start:i+1]
        previous = past[:-1] if len(past)>1 else past
        # No absolute position, time of day, course identifier, or future frame.
        temporal = [float(i-start),float(previous[:,1].mean()),float(previous[:,2].mean()),
                    float(np.mean(previous[:,1]-values[i,1])),float(previous[:,5].mean()),
                    float(previous[:,12].mean()),float(previous[:,1].max()),
                    float(np.sign(motion[start:i+1,2]*motion[i,2]).mean())]
        output[i] = np.r_[values[i],temporal]
    return output


def group(track,hashes):
    result = data(track,hashes)
    path = PARENT/f"{track}-observations.npz"
    receipt = path.with_suffix(".json")
    assert old.digest(path)==old.read(receipt)["arrays_sha256"]
    with np.load(path,allow_pickle=False) as raw:
        assert list(zip(raw["run"],raw["frame"],strict=True))==result["keys"]
        result["phase_x"] = frame_features(result["rows"],result["sections"],raw["motion"])
    result["phase_y"] = np.array([CLASS.get(r["truth"],-1) for r in result["rows"]])
    hashes.update((path,receipt,Path(__file__),Path(adjacent.__code__.co_filename)))
    return result
