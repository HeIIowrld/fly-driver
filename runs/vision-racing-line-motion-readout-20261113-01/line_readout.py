"""Read a local racing line from causal camera boundaries, body motion and FlyVis T4/T5."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from replay_corner_boundary_states import sequence
from score_corner_captures import adjacent, MIN_SPEED_MPS

HORIZONS = np.array([5., 10., 20.])
MODES = ("body", "boundary", "boundary_motion", "motion")
ALPHAS = (10., 100., 1000., 10000.)
WEIGHTS = (0., 1.)
CAMERA_STEP = .25
NATIVE_CADENCE = (.045, .18)


def metadata(sensor):
    return [dict(run=str(run), frame=int(frame), generation=str(gen), received_s=float(time))
            for run, frame, gen, time in zip(*(sensor[k] for k in ("run", "frame", "generation", "received_s")), strict=True)]


def sensor_features(sensor):
    rows = metadata(sensor)
    if not rows:
        raise ValueError("A nonempty sensor timeline is required")
    body, boundaries, visual, stations = (sensor[k] for k in ("motion", "sections", "t4t5", "stations"))
    moving = np.isfinite(body).all(axis=1) & (body[:,1] >= MIN_SPEED_MPS)
    consecutive = np.array([False, *(adjacent(a,b) for a,b in zip(rows, rows[1:]))])
    linked = consecutive[1:] & moving[1:] & moving[:-1]
    dt = np.diff(body[:,0])
    linked &= (dt > NATIVE_CADENCE[0]) & (dt <= NATIVE_CADENCE[1])
    aligned = sequence(stations, consecutive, boundaries, body)
    observed = np.isfinite(boundaries)
    geometric = observed.sum(axis=(1,2)) >= 4
    combined_visual = np.isfinite(visual).all(axis=(1,2))
    # Three observed samples must be continuous on both the camera and body clocks.
    motion = np.zeros(len(rows), dtype=bool)
    motion[2:] = linked[:-1] & linked[1:] & combined_visual[2:]
    history = np.zeros((len(rows), 3, 2), dtype=float)
    history[:,0] = np.where(moving[:,None], body[:,1:], 0.)
    history[1:,1] = np.where(linked[:,None], history[:-1,0], history[1:,0])
    history[2:,2] = np.where((linked[:-1]&linked[1:])[:,None], history[:-2,0], history[2:,1])
    history[1:2,2] = history[1:2,1]
    history[0,1:] = history[0,0]
    return dict(history=history, observed=boundaries, aligned=aligned, stations=stations,
                visual=visual, moving=moving, geometric=geometric, motion=motion)


def encoded(features, mode, sign):
    """Evaluate in either the original or the reflected vehicle coordinate system."""
    h = features["history"]
    speed = h[:,:,0]
    curvature = sign * h[:,:,1] / np.maximum(speed, 5.)
    body = np.column_stack((speed[:,0]/30, curvature*30, (speed[:,0]-speed[:,1])/8,
                            curvature[:,0]-curvature[:,1]))
    extra = []
    if mode in ("boundary", "boundary_motion"):
        for name in ("observed", "aligned"):
            e = features[name] if sign == 1. else -features[name][:,:,::-1]
            mask = np.isfinite(e)
            extra.extend((np.where(mask,e/5.,0.).reshape(len(e),-1), mask.reshape(len(e),-1)))
            pairs = mask.all(axis=2)
            middle = np.where(pairs,np.sum(e,axis=2)/2.,0.)
            jumps = np.diff(middle,axis=1)
            paired = pairs[:,1:] & pairs[:,:-1]
            bend = np.diff(jumps,axis=1)
            bend = np.where(paired[:,1:] & paired[:,:-1],bend,0.)
            extra.extend((middle/5., np.where(paired,jumps,0.), bend))
    if mode in ("motion", "boundary_motion"):
        # Each orientation keeps the existing pooled view, temporal and direction channels.
        extra.append(features["visual"][:,0 if sign == 1. else 1])
    return np.column_stack((body, *extra))


def availability(features, mode):
    available = features["moving"].copy()
    if mode in ("boundary", "boundary_motion"):
        available &= features["geometric"]
    if mode in ("motion", "boundary_motion"):
        available &= features["motion"]
    return available


def train(examples, mode, alpha, balance):
    columns, targets, weights = [], [], []
    for features, target in examples:
        use = availability(features, mode) & np.isfinite(target).all(axis=(1,2))
        if not use.any():
            raise ValueError("Development recording has no fully labeled sensor frames")
        y = target[use,:,1]
        weight = 1. + balance * np.minimum(abs(y[:,-1]), 4.)
        weight /= weight.sum()
        for sign in (1., -1.):
            columns.append(encoded(features,mode,sign)[use])
            targets.append(sign*y)
            weights.append(weight)
    x, y, w = (np.concatenate(parts) for parts in (columns, targets, weights))
    w *= len(w)/w.sum()
    scaler = StandardScaler().fit(x, sample_weight=w)
    x = np.clip(scaler.transform(x),-6.,6.)
    regressor = Ridge(alpha=alpha).fit(x, y, sample_weight=w)
    return scaler, regressor


def lateral_prediction(features, mode, model):
    use = availability(features, mode)
    predicted = np.full((len(use),len(HORIZONS)),np.nan)
    if use.any():
        scaler, readout = model
        plus, minus = (np.clip(scaler.transform(encoded(features,mode,s)[use]),-6.,6.) for s in (1.,-1.))
        predicted[use] = (readout.predict(plus)-readout.predict(minus))*.5
    return predicted


def bounded_path(lateral):
    """Build arc-spaced XY points with an origin and monotonically increasing x."""
    if not np.isfinite(lateral).all():
        return None
    arc = np.arange(0.,HORIZONS[-1]+CAMERA_STEP*.5,CAMERA_STEP)
    requested = PchipInterpolator(np.r_[0.,HORIZONS],np.r_[0.,lateral])(arc)
    # Limit lateral movement to 80% of arc length in every subinterval.
    dy = np.clip(np.diff(requested),-.8*CAMERA_STEP,.8*CAMERA_STEP)
    dx = np.sqrt(CAMERA_STEP**2-dy**2)
    path = np.column_stack((np.r_[0.,np.cumsum(dx)],np.r_[0.,np.cumsum(dy)]))
    return path


def predict(features, mode, model):
    lateral = lateral_prediction(features,mode,model)
    points = np.full((len(lateral),len(HORIZONS),2),np.nan)
    for i in np.flatnonzero(np.isfinite(lateral).all(axis=1)):
        points[i] = bounded_path(lateral[i])[(HORIZONS/CAMERA_STEP).astype(int)]
    return points


def constant_body_path(features):
    curvature = features["history"][:,0,1] / np.maximum(features["history"][:,0,0], 5.)
    curvature = np.clip(curvature, -.05, .05)
    arc_angle = curvature[:,None] * HORIZONS
    path = np.stack((HORIZONS*np.sinc(arc_angle/np.pi),
                     HORIZONS*np.sin(arc_angle/2)*np.sinc(arc_angle/(2*np.pi))),axis=2)
    path[~features["moving"]] = np.nan
    return path
