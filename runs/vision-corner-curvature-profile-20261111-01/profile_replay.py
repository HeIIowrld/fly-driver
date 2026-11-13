"""Predict road-curvature profiles from archived T4/T5 features."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from audit import INPUTS, RUNS
from evaluate_live_corner_geometry import load_edges, load_track_truth
from evaluate_fused_corner import frame_key, read_rows
from flybrain_drive.ac_camera_archive import CameraArchive
from pair_audit import CALIBRATION_TRACKS, TRACKS, load, sources
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import compare_tracks, score

OUT=Path(__file__).parent
OFFSETS=np.arange(0.,33.,2.)
CUTOFFS=(.10,.15,.20,.25,.35,.50,.65,1.0)
TRACK_ROOT=Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks")

def archive_path(track):
    if track=="imola":
        return RUNS/"vision-imola-approach-capture-20261111-02/archive-map.json"
    if track=="spa":
        return RUNS/"vision-spa-independent-capture-20261111-01/archive-map.json"
    return RUNS/INPUTS[track][1]

def track_reference(path):
    edges,_=load_edges(path)
    center=((edges[0]+edges[1])*.5)[:,[2,0]]
    diff=np.roll(center,-1,axis=0)-center
    lengths=np.linalg.norm(diff,axis=1)
    arc=np.r_[0.,np.cumsum(lengths)]
    length=arc[-1]
    count=int(np.ceil(length))
    grid=np.linspace(0,length,count,endpoint=False)
    step=length/count
    xy=np.column_stack([np.interp(grid,arc,np.r_[v,v[0]]) for v in center.T])
    smooth=gaussian_filter1d(xy,3./step,axis=0,mode="wrap")
    d=(np.roll(smooth,-1,axis=0)-np.roll(smooth,1,axis=0))/(2*step)
    dd=(np.roll(smooth,-1,axis=0)+np.roll(smooth,1,axis=0)-2*smooth)/(step*step)
    k=(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.maximum(.01,np.linalg.norm(d,axis=1))**3
    _,_,spans=load_track_truth(path)
    def targets(body):
        if not body or not body.get("physics_available"):
            return None
        p=np.asarray(body.get("renderer_only_position",[]))
        h=np.asarray(body.get("renderer_only_look",[]))
        if p.shape!=(3,) or h.shape!=(3,) or not np.isfinite([p,h]).all():
            return None
        p,h=p[[2,0]],h[[2,0]]
        t=np.clip(np.sum((p-center)*diff,axis=1)/lengths**2,0,1)
        delta=p-(center+t[:,None]*diff)
        j=int(np.argmin(np.sum(delta**2,axis=1)))
        if np.linalg.norm(delta[j])>8 or h@(diff[j]/lengths[j])<.7:
            return None
        station=(arc[j]+t[j]*lengths[j])%length
        query=(station+OFFSETS)%length
        curvature=np.interp(query,np.r_[grid,length],np.r_[k,k[0]]).clip(-.02,.02)/.02
        spans_profile=np.zeros_like(query)
        for start,end,side in spans:
            selected=(start<=query)&(query<end)
            if end>length:
                selected |= query<end-length
            spans_profile[selected]=side
        return curvature,spans_profile
    return targets

def labels(track,group,hashes):
    geometry=TRACK_ROOT/track/"ai/fast_lane.ai"
    mapping=archive_path(track)
    final,base,config=sources(track)
    proof=read(final/"result.json")
    assert digest(mapping)==read(base/"result.json")["archives_sha256"]
    pinned=read(Path(read(config)["base_protocol"]))["hashes"]
    assert pinned.get(str(geometry.resolve()))==digest(geometry)
    reference=track_reference(geometry)
    wanted={frame_key(row):i for i,row in enumerate(group["selected"])}
    by_kind=np.full((len(wanted),2,len(OFFSETS)),np.nan)
    for tag,folder in read(mapping).items():
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256==proof["camera_bundle_sha256_by_run"][tag]
            for line in archive.read("stream.jsonl").splitlines():
                capture=json.loads(line)
                i=wanted.get((tag,capture["capture_id"]))
                if i is None:
                    continue
                name=Path(capture["file"]).name
                assert archive.by_name[name]["sha256"]==capture["file_sha256"]
                meta=archive.json(name)
                assert meta["track_id"]==track and meta["index"]==capture["capture_id"]
                assert meta["generation"]==capture["generation"]
                view=max(meta["views"],key=lambda v:v["render_call_finished_clock"])
                value=reference(view.get("source_body_after_render"))
                if value is not None:
                    by_kind[i]=value
    hashes.update((geometry,mapping))
    return by_kind

def fit(groups,kind):
    data,target,weights=[],[],[]
    for group in groups:
        y=group["labels"][:,kind]
        truth=np.array([row["truth"] for row in group["selected"]])
        use=np.isfinite(y).all(axis=1)&np.isin(truth,("approach","straight","out_of_range","in_corner"))
        x=group["x"][use]
        y=y[use]
        approach=truth[use]=="approach"
        w=np.where(approach,.5/max(1,approach.sum()),.5/max(1,(~approach).sum()))
        data.append(x.reshape(-1,x.shape[-1]))
        target.append(np.stack((y,-y),axis=1).reshape(-1,y.shape[-1]))
        weights.append(np.repeat(w,2))
    x,y,w=np.concatenate(data),np.concatenate(target),np.concatenate(weights)
    w *= len(w)/w.sum()
    scaler=StandardScaler().fit(x,sample_weight=w)
    model=Ridge(alpha=500.).fit(np.clip(scaler.transform(x),-6,6),y,sample_weight=w)
    return scaler,model

def predict(group,fit):
    scaler,model=fit
    x=group["x"]
    prediction=model.predict(np.clip(scaler.transform(x.reshape(-1,x.shape[-1])),-6,6)).reshape(-1,2,len(OFFSETS))
    return .5*(prediction[:,0]-prediction[:,1])

def onset(profile,cut,approach_shape):
    sign=np.where(profile>=cut,1,np.where(profile<=-cut,-1,0))
    for i in range(2,11):
        if not sign[i] or sign[i+1]!=sign[i]:
            continue
        if approach_shape and (abs(profile[i-1])>=cut or abs(profile[i-2])>=cut):
            continue
        return (1 if sign[i]>0 else 2),float(OFFSETS[i])
    return None

def score_profiles(group,pred,cut,approach_shape):
    values=iter(pred)
    out=[]
    for row,ok in zip(group["rows"],group["valid"],strict=True):
        if ok:
            forecast=onset(next(values),cut,approach_shape) if cut is not None else None
            eligible=row["state"]=="straight" or row["state"]=="abstain" and row["abstain_reason"]=="presence_rejected"
            if forecast and eligible:
                row=score(row,forecast)
        out.append(row)
    return compare_tracks(group["original"],out,group["rules"])

def select(calibration,groups):
    ranked=[]
    for cut,shape in [(c,s) for c in CUTOFFS for s in (False,True)]+[(None,True)]:
        checks={k:score_profiles(groups[k],p,cut,shape) for k,p in calibration.items()}
        if any(report["metrics"]["candidate"]["outside"]>max(groups[k]["before"]["metrics"][name]["outside"]
               for name in ("parent","candidate")) for k,report in checks.items()):
            continue
        n=lambda key:sum(v["metrics"]["candidate"][key] for v in checks.values())
        ranked.append(((n("left"),n("joint"),-n("outside"),cut is None,shape,cut or 0.),(cut,shape)))
    return max(ranked)[1]

def diagnostics(group,pred,kind):
    targets=group["labels"][:,kind]
    use=np.isfinite(targets).all(axis=1)
    error=np.abs(pred[use]-targets[use])
    ideal=actual=matching=0
    for row,estimated,target in zip(group["selected"],pred,targets,strict=True):
        if row["truth"]!="approach" or not np.isfinite(target).all():
            continue
        cut=.10 if kind==0 else .50
        for key,values in (("target",target),("estimated",estimated)):
            forecast=onset(values,cut,True)
            valid=bool(forecast and str(forecast[0])==row["label"])
            if key=="target":
                ideal+=valid
            elif valid:
                actual+=1
                matching+=abs(forecast[1]-row["truth_distance_m"])<=5.
    return dict(available=int(use.sum()),scaled_mae=float(error.mean()),target_profile_direction_hits=ideal,
                predicted_profile_direction_hits=actual,predicted_profile_joint_hits=matching)

def run():
    hashes={Path(__file__),Path("runs/vision-corner-matched-approaches-20261111-01/pair_audit.py"),
            Path("scripts/evaluate_live_corner_geometry.py")}
    groups={}
    for track in TRACKS:
        group=load(track,hashes)
        group["labels"]=labels(track,group,hashes)
        groups[track]=group
        print(track,"registered",int(np.isfinite(group["labels"]).all(axis=(1,2)).sum()),flush=True)
    reports={}
    for kind,name in enumerate(("curvature","span")):
        fits={keys:fit([groups[k] for k in keys],kind) for count in (3,4,5)
              for keys in combinations(CALIBRATION_TRACKS,count)}
        reports[name]={}
        for track,g in groups.items():
            train=tuple(k for k in CALIBRATION_TRACKS if k!=track)
            pred=predict(g,fits[train])
            calibration={k:predict(groups[k],fits[tuple(j for j in train if j!=k)]) for k in train}
            cut,shape=select(calibration,groups)
            comparison=score_profiles(g,pred,cut,shape)
            diagnostic=diagnostics(g,pred,kind)
            sweep={f"{cut}:{shape}":score_profiles(g,pred,cut,shape)["metrics"]["candidate"]
                   for cut in CUTOFFS for shape in (False,True)}
            reports[name][track]=dict(cut=cut,approach_shape=shape,comparison=comparison,
                                     profile=diagnostic,diagnostic_sweep=sweep)
            print(name,track,cut,shape,diagnostic,comparison["metrics"]["candidate"],flush=True)
    write(OUT/"result.json",dict(tracks=reports,offsets_m=OFFSETS.tolist(),seed=None,
          source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="causal_profile_development_comparison_complete"))

if __name__=="__main__":
    run()
