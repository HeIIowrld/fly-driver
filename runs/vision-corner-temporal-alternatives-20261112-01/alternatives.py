"""Generate alternative bends and associate them with earlier local observations."""
from pathlib import Path
import json
import numpy as np

from observations import OUT, GATE, TRACKS, DEVELOP, SILVER, VAL, old
from features import local_features, load_silver
from evaluate_vallelunga import PREVIOUS as VAL_REPLAY, features as val_features
from evaluate_shape import candidate_from_head
from score_corner_captures import adjacent
from corner_fusion_boundaries import transform, advance

MAX_LAG = 3
MAX_ALONG = 3.
MAX_LATERAL = 2.5
CUT = .002


def segments(boundaries,radius):
    """Enumerate the first sample of every supported consecutive signed bend."""
    stations = old.STATIONS
    width = boundaries[:,1]-boundaries[:,0]
    support = np.isfinite(boundaries).all(axis=1)&(width>3.)&(width<20.)
    profile = {}
    for center in np.arange(4.,22.):
        relative = stations-center
        mask = support&(abs(relative)<=radius)
        x = relative[mask]
        if len(x)<4 or x[0]>=0. or x[-1]<=0. or max(np.diff(x))>4.:
            continue
        design = np.column_stack((np.ones(len(x)),x,x*x))
        coeff = np.linalg.lstsq(design,boundaries[mask],rcond=None)[0]
        k = 2.*coeff[2]/(1.+coeff[1]**2)**1.5
        profile[center] = float(np.copysign(min(abs(k)),k[0])) if k[0]*k[1]>0. else 0.
    candidates = []
    previous_side, previous_end = 0, None
    for start in range(4,21):
        a,b = profile.get(start,0.), profile.get(start+1,0.)
        if a*b<=0. or min(abs(a),abs(b))<CUT:
            continue
        side = 1 if a>0. else 2
        if previous_side!=side or previous_end!=start:
            candidates.append((side,float(start),min(abs(a),abs(b))))
        previous_side,previous_end = side,start+1
    return candidates


def onset_point(boundaries,distance):
    valid = np.isfinite(boundaries).all(axis=1)
    x = old.STATIONS[valid]
    y = boundaries[valid].mean(axis=1)
    i = np.searchsorted(x,distance)
    if i<len(x) and x[i]==distance:
        return np.array([distance,y[i]])
    if 0<i<len(x) and x[i]-x[i-1]<=4.:
        return np.array([distance,np.interp(distance,x[i-1:i+1],y[i-1:i+1])])
    return None


def alternatives(boundaries,head,primary):
    options = {}
    for j,radius in enumerate((10.,6.)):
        for side,distance,strength in segments(boundaries,radius):
            flags = options.setdefault((side,distance),np.zeros(3))
            flags[j] = strength
    wide = [key for key,values in options.items() if values[0]>0.]
    assert (wide[0] if wide else None) == primary
    model = candidate_from_head(head)
    if model:
        options.setdefault(model,np.zeros(3))[2] = head[model[0]]
    return [dict(offer=key,source=values,point=onset_point(boundaries,key[1]))
            for key,values in options.items()]


def history_vector(candidate, older, steps):
    """Only advance earlier 2D points through verified, causal body steps."""
    output = []
    current = candidate["point"]
    matches = 0
    for observations, chain in zip(older,steps,strict=True):
        aligned = []
        if current is not None and chain is not None:
            for previous in observations:
                if previous["offer"][0]!=candidate["offer"][0] or previous["point"] is None:
                    continue
                point = previous["point"]
                for rotation, move in chain:
                    point = advance(point,rotation,move)
                residual = current-point
                aligned.append((float(np.linalg.norm(residual/[MAX_ALONG,MAX_LATERAL])),residual))
        if not aligned:
            output.extend((0.,0.,0.,0.))
            continue
        _, difference = min(aligned,key=lambda entry:entry[0])
        consistent = abs(difference[0])<=MAX_ALONG and abs(difference[1])<=MAX_LATERAL
        matches += int(consistent)
        output.extend((1.,float(consistent),*np.clip(difference,[-12.,-10.],[12.,10.])))
    output.append(float(matches))
    return output


def sequence(rows,raw):
    motion, bounds = raw["motion"],raw["sections"]
    lists, incoming, matched = [],[],[]
    for i,row in enumerate(rows):
        primary = tuple(raw["primary"][i]) if np.isfinite(raw["primary"][i]).all() else None
        usable = np.isfinite(motion[i]).all() and motion[i,1]>=old.MIN_SPEED_MPS
        if usable:
            values = alternatives(bounds[i],raw["heads"][i],primary)
        else:
            assert primary is None
            values = []
        lists.append(values)
        step = None
        if (i and adjacent(rows[i-1],row) and np.isfinite(motion[i-1:i+1]).all()
            and min(motion[i-1:i+1,1])>=old.MIN_SPEED_MPS):
            step = transform(motion[i],motion[i-1])
        incoming.append(step)
        older, chains = [],[]
        for lag in range(1,MAX_LAG+1):
            chain = incoming[max(0,i-lag+1):i+1]
            connected = i>=lag and len(chain)==lag and all(s is not None for s in chain)
            older.append(lists[i-lag] if connected else [])
            chains.append(chain if connected else None)
        matched.append([history_vector(c,older,chains) for c in values])
    return lists,matched


def load_features(track,hashes):
    entry = old.context()
    runner = entry.import_replay()
    if track==VAL:
        previous = GATE/"vallelunga-result.json"
        proof = old.read(previous)
        admission = VAL_REPLAY/"admission/rows.jsonl"
        assert old.digest(admission)==old.read(VAL_REPLAY/"admission/result.json")["rows_sha256"]
        truth = [json.loads(line) for line in admission.read_text(encoding="utf8").splitlines()]
        hashes.update((admission,previous,Path(val_features.__code__.co_filename)))
        receipt = old.read(OUT/f"{track}-observations.json")
        return val_features(VAL_REPLAY/"features",truth,receipt["camera_bundle_sha256_by_run"],hashes)
    loader = load_silver if track==SILVER else runner.profile.load
    hashes.add(Path(loader.__code__.co_filename))
    return loader(hashes) if track==SILVER else loader(track,hashes)


def data(track,hashes):
    if track not in TRACKS:
        raise ValueError("Unknown circuit")
    group = load_features(track,hashes)
    path = OUT/f"{track}-observations.npz"
    receipt = path.with_suffix(".json")
    assert old.digest(path)==old.read(receipt)["arrays_sha256"]
    hashes.update((path,receipt,Path(__file__),Path(transform.__code__.co_filename),
                   Path(candidate_from_head.__code__.co_filename),Path(local_features.__code__.co_filename),
                   Path(adjacent.__code__.co_filename)))
    with np.load(path,allow_pickle=False) as data:
        keys = list(zip(data["run"],data["frame"],strict=True))
        raw = {name:data[name] for name in ("motion","sections","primary","edge_base","heads")}
    rows = group["rows"]
    assert keys==[(r["run"],r["frame"]) for r in rows]
    by_key = {(r["run"],r["frame"]):x for r,x in zip(group["selected"],group["x"],strict=True)}
    pools,temporal = sequence(rows,raw)
    flat_index, offers, edge, history, motion, error, origin = [],[],[],[],[],[],[]
    for i,(row,candidates,past) in enumerate(zip(rows,pools,temporal,strict=True)):
        visual = by_key.get(keys[i])
        if visual is None or not np.isfinite(visual).all():
            continue
        for c,h in zip(candidates,past,strict=True):
            p = c["offer"]
            flat_index.append(i)
            offers.append(p)
            edge.append(np.r_[local_features(old.STATIONS,raw["sections"][i],p,None,0.)[:-4],c["source"]])
            history.append(h)
            motion.append(visual[p[0]-1])
            error.append(old.category(row,p))
            origin.append(c["source"])
    return dict(rows=rows,keys=keys,primary=raw["primary"],edge_base=raw["edge_base"],sections=raw["sections"],
                pools=[[c["offer"] for c in p] for p in pools],index=np.asarray(flat_index,dtype=int),
                offers=offers,edge=np.asarray(edge),history=np.asarray(history),motion=np.asarray(motion),
                errors=np.asarray(error),target=np.asarray(error)=="joint_hit",origin=np.asarray(origin))
