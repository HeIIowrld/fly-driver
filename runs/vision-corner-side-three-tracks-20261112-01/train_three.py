"""Test whether adding Imola improves dense side-edge selection across circuits."""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
DENSE=OUT.parent/"vision-corner-side-heatmap-20261112-01"
EXTRA=OUT.parent/"vision-corner-extra-side-labels-20261112-01"
sys.path.insert(0,str(DENSE))
from train_heat import (CROP,SIZE,CACHE,TEST,VAL,SEED,old,baseline_data,crops,pixel_targets,
                        candidates,fit,infer,probability,pick,sampling,RAW_ARCHIVE,ROUTING,CAPTURE,
                        MESH,metrics,mesh_score,load_observer,observe,frames,SideHeat)
from train_heat import arrays as previous_arrays

DEV=("magione","spa","imola")
MODES=("contrast","heat","heat_and_contrast","gain_005","gain_010")
PAIR_MODES=("contrast","two_track","three_track","selected")


def imola(sources):
    track="imola"
    label_path=EXTRA/f"{track}-train-labels.npz"
    receipt=label_path.with_suffix(".json")
    proof=old.read(receipt)
    assert old.digest(label_path)==proof["labels_sha256"]
    for name,sha in proof["source_sha256"].items():
        assert old.digest(Path(name))==sha
    group,_,mapping,bundles=baseline_data(track,sources)
    keys=[(r["run"],r["frame"]) for r in group["rows"]]
    assert bundles==proof["camera_bundle_sha256_by_run"]
    with np.load(label_path,allow_pickle=False) as d:
        assert keys==list(zip(d["run"],d["frame"],strict=True))
        assert np.array_equal(d["stations"],old.STATIONS)
        assert d["views"].tolist()==["front","left","right","rear"]
        pixels=np.stack((d["image_px"][:,2,:,0],d["image_px"][:,1,:,1]),axis=1).copy()
        lateral=d["lateral_m"]
        witness=d["adjacent_invalid"]
    pixels[:,1,:,0]=319.-pixels[:,1,:,0]
    visible=witness.transpose(0,2,1)&np.isfinite(pixels).all(-1)
    visible&=(pixels[...,1]>=CROP[0]+1)&(pixels[...,1]<CROP[1]-1)
    images,available=crops(track,mapping,bundles,keys)
    flat=pixels.reshape(-1,len(old.STATIONS),2)
    mask=visible.reshape(-1,len(old.STATIONS))
    xy=np.stack((flat,np.full_like(flat,np.nan)),axis=2)
    marks,weights=pixel_targets(xy,np.stack((mask,np.zeros_like(mask)),axis=-1))
    usable=(mask.sum(axis=1)>=2)&np.repeat(available,2)
    data_path=CACHE/f"{track}-sections.npz"
    cached=data_path.with_suffix(".json")
    assert old.digest(data_path)==old.read(cached)["sections_sha256"]
    with np.load(data_path,allow_pickle=False) as d:
        assert keys==list(zip(d["run"],d["frame"],strict=True))
        observed=d["values"][:,0]
        assert np.array_equal(d["stations"],old.STATIONS[:observed.shape[1]])
    sections=np.full((len(keys),len(old.STATIONS),2),np.nan)
    sections[:,:observed.shape[1]]=observed
    options=list(candidates(track,mapping,bundles,keys,sections))
    sources.update((label_path,receipt,data_path,cached,mapping))
    return dict(x=images.reshape(-1,SIZE[1],SIZE[0],3)[usable],y=marks[usable,0],
                weight=weights[usable,0],images=images,options=options,truth=lateral,
                eligible=visible.transpose(0,2,1))


def choose(mode,heat,index,grid,x,lut):
    if not mode.startswith("gain_"):
        return pick(mode,heat,index,grid,x,lut)
    p=probability(heat,index,grid,lut)
    old_index=int(np.argmax(x[:,0]))
    new_index=int(np.argmax(p+.15*(x[:,0]-x[:,0].max())/max(np.ptp(x[:,0]),1.)))
    gain=int(mode.removeprefix("gain_"))/100.
    return grid[new_index if p[new_index]-p[old_index]>=gain else old_index]


def score(group,heat,lut):
    truth,visible=group["truth"],group["eligible"]
    rows=[(i,grid,x) for i,grid,x in group["options"] if visible[i]]
    result={}
    for mode in MODES:
        delta=np.array([abs(choose(mode,heat,i,grid,x,lut))-abs(truth[i]) for i,grid,x in rows])
        result[mode]=dict(points=len(delta),mae_m=float(abs(delta).mean()),
                          signed_m=float(delta.mean()),within_1_25_m=int((abs(delta)<=1.25).sum()))
    return result


def patch(options,raw,mode,heat,lut):
    sections=raw.copy()
    for index,grid,x in options:
        assert not np.isfinite(raw[index])
        sections[index]=choose(mode,heat,index,grid,x,lut)
    width=sections[:,:,1]-sections[:,:,0]
    invalid=np.isfinite(width)&((width<=3.)|(width>=20.))
    sections[invalid]=raw[invalid]
    np.testing.assert_array_equal(sections[np.isfinite(raw)],raw[np.isfinite(raw)])
    return sections


def replay(track,name,mapping,new_model,old_model,device,choice,lut,observers,settings,pins):
    directory=RAW_ARCHIVE/name
    raw_path=directory/"sensors.npz"
    receipt_path=directory/"result.json"
    receipt=old.read(receipt_path)
    assert old.digest(raw_path)==receipt["inputs_sha256"]
    with np.load(raw_path,allow_pickle=False) as d:
        raw={k:d[k] for k in d.files}
    keys=list(zip(raw["run"],raw["frame"],strict=True))
    images,valid=crops(track,mapping,receipt["archive_bundles"],keys)
    options=list(candidates(track,mapping,receipt["archive_bundles"],keys,raw["sections"],raw["motion"]))
    assert all(valid[i[0]] for i,_,_ in options)
    a,b=(infer(m,images,device) for m in (old_model,new_model))
    reference_path=ROUTING/f"{track}-state-distance.jsonl"
    route=old.read(ROUTING/"straight-verification.json")
    reference=frames(reference_path,route["tracks"][track]["rows_sha256"])
    prior=old.read(DENSE/"result.json")["tracks"][track]["methods"]
    methods={}
    for label,mode,heat in zip(PAIR_MODES,("contrast","heat_and_contrast","heat_and_contrast",choice),(b,a,b,b),strict=True):
        sections=patch(options,raw["sections"],mode,heat,lut)
        sensor=OUT/f"{track}-{label}-sensors.npz"
        np.savez_compressed(sensor,**dict(raw,sections=sections))
        with np.load(sensor,allow_pickle=False) as data:
            predictions,aligned,_=observe(data,observers,settings)
        trace=OUT/f"{track}-{label}-frames.jsonl"
        old.jsonl(trace,predictions)
        scores,hits=metrics(predictions,reference)
        entry=dict(mode=mode,metrics=scores,right_hits=hits,frames_sha256=old.digest(trace),
                   sensors_sha256=old.digest(sensor))
        if label in ("contrast","two_track"):
            old_label="contrast" if label=="contrast" else "heat_and_contrast"
            assert entry["frames_sha256"]==prior[old_label]["frames_sha256"]
        if track==TEST:
            with np.load(directory/"predictions/boundaries.npz",allow_pickle=False) as base:
                entry["mesh"]=mesh_score(aligned,base["estimated"],reference,keys)
        methods[label]=entry
    sources={*pins,raw_path,receipt_path,reference_path,mapping,DENSE/"result.json"}
    if track==TEST:
        sources.update((MESH,MESH.with_suffix(".json"),directory/"predictions/boundaries.npz"))
    return dict(methods=methods,source_sha256={str(p):old.digest(p) for p in sorted(sources)})


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut=sampling(old.read(old.MOUNT))
    sources={Path(__file__),old.MOUNT,Path(candidates.__code__.co_filename),Path(fit.__code__.co_filename),
             Path(pixel_targets.__code__.co_filename),Path(crops.__code__.co_filename)}
    groups={}
    for t in DEV:
        groups[t]=imola(sources) if t=="imola" else previous_arrays(t,sources)
        print("Prepared",t,len(groups[t]["x"]),flush=True)
    folds={}
    for held,group in groups.items():
        model,details=fit([v for k,v in groups.items() if k!=held],device)
        folds[held]=score(group,infer(model,group["images"],device),lut)
        print("Fold",held,folds[held],flush=True)
    def risk(mode):
        ratio=[folds[t][mode]["mae_m"]/folds[t]["contrast"]["mae_m"] for t in DEV]
        return max(ratio),sum(ratio)
    choice=min(MODES,key=risk)
    model,details=fit(list(groups.values()),device)
    checkpoint=OUT/"side-three.pt"
    torch.save(model.state_dict(),checkpoint)
    frozen=OUT/"training.json"
    assert not any(TEST in str(p) or VAL in str(p) for p in sources)
    old.write(frozen,dict(status="frozen_before_replay",development=DEV,excluded=[TEST,VAL],choice=choice,
              seed=SEED,source_revision=old.revision(),folds=folds,fit=details,checkpoint_sha256=old.digest(checkpoint),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="three_circuit_side_learning"))
    print("Frozen",choice,flush=True)
    old_path=DENSE/"side-heat.pt"
    assert old.digest(old_path)==old.read(DENSE/"training.json")["checkpoint_sha256"]
    old_model=SideHeat().to(device)
    old_model.load_state_dict(torch.load(old_path,map_location=device,weights_only=True))
    old_model.eval()
    observers,settings,pins=load_observer()
    pins.update((old_path,checkpoint,frozen,DENSE/"training.json"))
    result={}
    for t,n,mapping in (
            (TEST,"silverstone",CAPTURE),
            (VAL,"vallelunga",OUT.parent/"vision-vallelunga-local-corner-capture-20261112-01"/"archive-map.json")):
        result[t]=replay(t,n,mapping,model,old_model,device,choice,lut,observers,settings,pins)
        print(t,{m:(r["right_hits"],r.get("mesh",{}).get("new_mae_m")) for m,r in result[t]["methods"].items()},flush=True)
    old.write(OUT/"result.json",dict(status="three_domain_archive_replay",training_sha256=old.digest(frozen),
             tracks=result,termination="two_and_three_circuit_models_compared"))


if __name__=="__main__":
    main()
