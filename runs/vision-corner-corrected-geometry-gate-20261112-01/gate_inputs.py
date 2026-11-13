"""Build candidate-confirmation examples from archived and offline-corrected geometry."""
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
EXTRA=OUT.parent/"vision-corner-extra-side-labels-20261112-01"
sys.path.insert(0,str(ROOT/"scripts"))
from replay_corner_boundary_states import (load_observer, old, sensor_rows, proposals, rank_candidates,
    frame_features, phase_scores, sequence, alt, causal_prefix, relative_features)
from directional_features import PARENT
from rank import encode
from train_phase import fit_phase
from routing import DEVELOP,SILVER,VAL


def read_development(track,sources,settings):
    path=PARENT/f"{track}-observations.npz"
    receipt=path.with_suffix(".json")
    proof=old.read(receipt)
    assert old.digest(path)==proof["arrays_sha256"]
    group=alt.load_features(track,sources)
    truth=group["rows"]
    keys=[(r["run"],r["frame"]) for r in truth]
    with np.load(path,allow_pickle=False) as data:
        assert keys==list(zip(data["run"],data["frame"],strict=True))
        sensors={k:data[k] for k in ("run","frame","sections","motion","heads")}
    sensors.update(stations=old.STATIONS, generation=np.array([r["generation"] for r in truth]),
                   received_s=np.array([r["received_s"] for r in truth]))
    by_key={(r["run"],r["frame"]):x for r,x in zip(group["selected"],group["x"],strict=True)}
    missing=np.full((2,settings["motion_features"]),np.nan)
    sensors["t4t5"]=np.stack([by_key.get(k,missing) for k in keys])
    sources.update((path,receipt))
    return sensors,truth,proof


def corrected(track,sensors,observation_proof,sources):
    path=EXTRA/f"{track}-train-labels.npz"
    if not path.is_file():
        return None
    receipt=path.with_suffix(".json")
    proof=old.read(receipt)
    assert proof["status"]=="offline_only" and old.digest(path)==proof["labels_sha256"]
    assert observation_proof["camera_bundle_sha256_by_run"]==proof["camera_bundle_sha256_by_run"]
    with np.load(path,allow_pickle=False) as data:
        for key in ("run","frame","stations"):
            np.testing.assert_array_equal(data[key],sensors[key])
        eligible=data["adjacent_invalid"] & np.isfinite(data["image_px"]).all(axis=-1).any(axis=1)
        edge=data["lateral_m"]
        assert np.isfinite(edge[eligible]).all()
    observed=sensors["sections"]
    updated=np.where(eligible,edge,observed)
    width=updated[:,:,1]-updated[:,:,0]
    invalid=np.isfinite(width)&((width<=3.)|(width>=20.))
    updated[invalid]=observed[invalid]
    applied=int((eligible&~invalid[:,:,None]).sum())
    sources.update((path,receipt))
    return dict(sensors,sections=updated),applied


class SensorArrays(dict):
    @property
    def files(self):
        return self.keys()


def build(sensors,settings,truth=None):
    data=SensorArrays(sensors)
    rows=sensor_rows(data,settings)
    body,raw,heads,visual=(data[k] for k in ("motion","sections","heads","t4t5"))
    contiguous=[False,*(alt.adjacent(a,b) for a,b in zip(rows,rows[1:]))]
    aligned=sequence(old.STATIONS,contiguous,raw,body)
    phase=frame_features(rows,raw,body)
    direct=proposals(rows,raw,body,heads)
    recovered=proposals(rows,aligned,body,heads)
    geometry,history,motion,offers,index,errors,origin=[],[],[],[],[],[],[]
    if truth is not None:
        assert [(r["run"],r["frame"]) for r in rows]==[(r["run"],r["frame"]) for r in truth]
    for i,((a,ah),(b,bh)) in enumerate(zip(zip(*direct),zip(*recovered),strict=True)):
        known={c["offer"] for c in a}
        options=[(c,h,raw[i]) for c,h in zip(a,ah,strict=True)]
        options.extend((c,h,aligned[i]) for c,h in zip(b,bh,strict=True) if c["offer"] not in known)
        if not options:
            continue
        local=causal_prefix(rows,body,i)
        for candidate,past,sections in options:
            offer=candidate["offer"]
            base=alt.local_features(old.STATIONS,sections,offer,None,0.)[:-4]
            direction=relative_features(old.STATIONS,sections,body[i],local,offer)
            geometry.append(np.r_[base,candidate["source"],phase[i],direction])
            history.append(past)
            motion.append(visual[i,offer[0]-1])
            index.append(i)
            offers.append(offer)
            origin.append(offer in known)
            if truth is not None:
                errors.append(old.category(truth[i],offer))
    group=dict(edge=np.asarray(geometry),history=np.asarray(history),motion=np.asarray(motion),
               index=np.asarray(index),offers=offers,phase_x=phase,original=np.asarray(origin),
               phase_y=np.asarray([{"straight":0,"out_of_range":0,"approach":1,"in_corner":2}.get(r["truth"],-1)
                                   for r in truth]) if truth is not None else None)
    if truth is not None:
        assert np.isfinite(group["motion"]).all(), "Development camera features absent on a candidate"
        group.update(errors=np.asarray(errors),target=np.asarray(errors)=="joint_hit")
    return dict(sensors=data,rows=rows,group=group,phase=phase,aligned=aligned,direct=direct,recovered=recovered)


def pick(built,chooser,phase_model):
    data=built["sensors"]
    offers,raw=rank_candidates(built["rows"],built["aligned"],data["sections"],data["motion"],data["t4t5"],
                               built["phase"],built["direct"],built["recovered"],chooser)
    phase=phase_scores(built["group"],phase_model)[:,2]
    lookup={(i,offer):k for k,(i,offer) in enumerate(zip(built["group"]["index"],built["group"]["offers"],strict=True))}
    selected=np.array([lookup[i,offer] for i,offer in offers.items()],dtype=int)
    return offers,raw,phase,selected


def scores(group,selected,models):
    subset={key:group[key][selected] for key in ("edge","history","motion")}
    valid=np.isfinite(subset["motion"]).all(axis=1)
    result={}
    for mode in ("temporal","t4t5"):
        include=valid if mode=="t4t5" else np.ones(len(selected),bool)
        p=np.full(len(selected),np.nan)
        if include.any():
            used={key:val[include] for key,val in subset.items()}
            p[include]=models[mode]["model"].predict_proba(encode(used,models[mode],mode))[:,1]
        result[mode]=p
    return result
