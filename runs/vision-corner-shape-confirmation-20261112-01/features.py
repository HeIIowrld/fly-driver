"""Candidate-local geometry and fixed motion features with a common admission mask."""
from pathlib import Path
import json

import numpy as np

from collect import OUT, SHAPE, DEVELOP, old
from evaluate_shape import SILVER
from heads import load_silver
from score_corner_captures import adjacent

ELIGIBLE = {"approach","straight","out_of_range","in_corner"}
OFFSETS = (-5.,0.,5.)


def local_features(stations, boundaries, offer, previous, dt):
    direction, distance = offer
    oriented = boundaries if direction==1 else -boundaries[:,::-1]
    good = np.isfinite(oriented).all(axis=1)
    width = oriented[:,1]-oriented[:,0]
    good &= (width>3.)&(width<20.)
    result = [distance, good.sum(), np.mean(width[good]) if good.any() else 0.,
              np.std(width[good]) if good.any() else 0.]
    for shift in OFFSETS:
        along = stations-distance-shift
        keep = good & (abs(along)<=8.)
        x, points = along[keep], oriented[keep]
        result.extend((len(x), float(np.mean(width[keep])) if len(x) else 0.))
        if len(x)<4:
            result.extend([0.]*5)
            continue
        design = np.column_stack((np.ones(len(x)),x,x*x))
        coeff = np.linalg.lstsq(design,points,rcond=None)[0]
        curvature = 2.*coeff[2]/(1.+coeff[1]**2)**1.5
        residual = np.sqrt(np.mean((design@coeff-points)**2))
        result.extend((*curvature, *coeff[1], residual))
    same = previous is not None and int(previous[0]) == direction
    result.extend((float(previous is not None),float(same),
                   (previous[1]-distance) if same else 0., dt if previous is not None else 0.))
    return np.asarray(result,dtype=float)


def cached(track, hashes):
    if track == SILVER:
        receipt = SHAPE/"result.json"
        proof = old.read(receipt)
        path = SHAPE/"cross-sections.npz"
        frames = SHAPE/"silverstone-frames.jsonl"
        assert old.digest(path)==proof["sections_sha256"] and old.digest(frames)==proof["rows_sha256"]
        hashes.update((receipt,path,frames))
        rows = [json.loads(line) for line in frames.read_text(encoding="utf8").splitlines()]
        offers = [r["offers"].get("shape_geometry") for r in rows]
        keys = [(r["run"],r["frame"]) for r in rows]
        base = [r["decisions"]["edge_base"] for r in rows]
        with np.load(path,allow_pickle=False) as data:
            assert list(zip(data["run"],data["frame"],strict=True))==keys
            sections = data["shape_geometry"]
        return keys, offers, sections, base
    path = OUT/f"{track}-candidates.npz"
    receipt = path.with_suffix(".json")
    proof = old.read(receipt)
    assert proof["track"]==track and track not in proof["shape_training"]
    assert old.digest(path)==proof["arrays_sha256"]
    hashes.update((path,receipt))
    with np.load(path,allow_pickle=False) as data:
        keys = list(zip(data["run"],data["frame"],strict=True))
        sections = data["sections"]
        offers = [(int(p[0]),float(p[1])) if np.isfinite(p).all() else None for p in data["candidates"]]
    return keys, offers, sections, [None]*len(keys)


def data(track, hashes):
    if track not in (*DEVELOP,SILVER):
        raise ValueError("Unknown circuit")
    entry = old.context()
    runner = entry.import_replay()
    loader = load_silver if track==SILVER else runner.profile.load
    hashes.update((Path(__file__),Path(loader.__code__.co_filename),Path(adjacent.__code__.co_filename)))
    group = loader(hashes) if track==SILVER else loader(track,hashes)
    keys, offers, sections, base = cached(track,hashes)
    return assemble(group, keys, offers, sections, base)


def assemble(group, keys, offers, sections, base):
    key = lambda r: (r["run"],r["frame"])
    rows = group["rows"]
    assert [key(r) for r in rows]==keys
    available = {key(r):x for r,x in zip(group["selected"],group["x"],strict=True)}
    edge, motion, index, target, errors = [], [], [], [], []
    for i,(row,offer) in enumerate(zip(rows,offers,strict=True)):
        if offer is None or keys[i] not in available:
            continue
        feature = available[keys[i]][int(offer[0])-1]
        if not np.isfinite(feature).all():
            continue
        has_previous = i>0 and adjacent(rows[i-1],row)
        previous = offers[i-1] if has_previous else None
        dt = row["received_s"]-rows[i-1]["received_s"] if has_previous else 0.
        edge.append(local_features(old.STATIONS,sections[i],offer,previous,dt))
        motion.append(feature)
        index.append(i)
        error = old.category(row,offer)
        target.append(error=="joint_hit")
        errors.append(error)
    return dict(rows=rows,keys=keys,offers=offers,base=base,index=np.array(index,dtype=int),
                edge=np.asarray(edge),motion=np.asarray(motion),target=np.asarray(target,dtype=bool),
                errors=np.asarray(errors))
