"""Train a dense outward-camera boundary heatmap on physically projected labels."""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT=Path(__file__).resolve().parent
CHAINS=OUT.parent/"vision-corner-side-chain-20261112-01"
sys.path.insert(0,str(CHAINS))
from trace_chains import (CACHE, DEV, TEST, VAL, SEED, old, labels, baseline_data,
                          candidates, RAW_ARCHIVE, ROUTING, CAPTURE, MESH, metrics, mesh_score,
                          load_observer, observe, frames)
from rank_side_pairs import sampling, stream, image, SIDES, STATIONS, DISTANCES
from model import CROP, SIZE, block, pixel_targets, input_image

MODES=("contrast","heat","heat_and_contrast")
EPOCHS=7
BATCH=64


class SideHeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(block(3,16,2),block(16,24,1),block(24,32,2),
                                    block(32,32,1),block(32,32,1),nn.Conv2d(32,1,1))

    def forward(self,x):
        y=self.features(x)
        return F.interpolate(y,size=(SIZE[1],SIZE[0]),mode="bilinear",align_corners=False)[:,0]


def crops(track,mapping,bundles,keys):
    index={key:i for i,key in enumerate(keys)}
    seen=set()
    images=np.zeros((len(keys),2,SIZE[1],SIZE[0],3),np.uint8)
    valid=np.zeros(len(keys),bool)
    for name,folder in old.read(mapping).items():
        for archive,item,_,views in stream(folder,bundles[name],track):
            key=name,item["capture_id"]
            if key not in index:
                continue
            i=index[key]
            assert i not in seen
            seen.add(i)
            if views:
                for side,view in enumerate(SIDES):
                    bitmap=image(archive,views[view])
                    assert bitmap.shape==(180,320,3)
                    if side:
                        bitmap=bitmap[:,::-1]
                    images[i,side]=cv2.resize(bitmap[CROP[0]:CROP[1]],SIZE,interpolation=cv2.INTER_AREA)
                valid[i]=True
    assert len(seen)==len(keys)
    return images,valid


def arrays(track,sources):
    truth,label_path=labels(track)
    group,_,mapping,bundles=baseline_data(track,sources)
    keys=[(r["run"],r["frame"]) for r in group["rows"]]
    assert keys==list(zip(truth["run"],truth["frame"],strict=True))
    pixels=np.stack((truth["image_px"][:,2,:,0],truth["image_px"][:,1,:,1]),axis=1).copy()
    pixels[:,1,:,0]=319.-pixels[:,1,:,0]
    visible=truth["adjacent_invalid"].transpose(0,2,1)&np.isfinite(pixels).all(-1)
    visible &= (pixels[...,1]>=CROP[0]+1)&(pixels[...,1]<CROP[1]-1)
    images,available=crops(track,mapping,bundles,keys)
    flat=pixels.reshape(-1,len(old.STATIONS),2)
    dummy=np.full_like(flat,np.nan)
    truth_xy=np.stack((flat,dummy),axis=2)
    mask=np.stack((visible.reshape(-1,len(old.STATIONS)),np.zeros_like(visible.reshape(-1,len(old.STATIONS)))),axis=-1)
    marks,weights=pixel_targets(truth_xy,mask)
    images_flat=images.reshape(-1,SIZE[1],SIZE[0],3)
    usable=(mask[:,:,0].sum(axis=1)>=2)&np.repeat(available,2)
    assert usable.any()
    path=CACHE/f"{track}-sections.npz"
    proof=path.with_suffix(".json")
    assert old.digest(path)==old.read(proof)["sections_sha256"]
    with np.load(path,allow_pickle=False) as data:
        assert keys==list(zip(data["run"],data["frame"],strict=True))
        baseline=data["values"][:,0]
    observed=np.full((len(keys),len(old.STATIONS),2),np.nan)
    observed[:,:baseline.shape[1]]=baseline
    options=list(candidates(track,mapping,bundles,keys,observed))
    sources.update((label_path,label_path.with_suffix(".json"),path,proof,mapping))
    return dict(x=images_flat[usable],y=marks[usable,0],weight=weights[usable,0],images=images,
                options=options,truth=truth["lateral_m"],eligible=visible.transpose(0,2,1))


def fit(groups,device):
    torch.manual_seed(SEED)
    model=SideHeat().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    images,marks,weights=(np.concatenate([g[k] for g in groups]) for k in ("x","y","weight"))
    rng=np.random.default_rng(SEED)
    losses=[]
    for epoch in range(EPOCHS):
        total=0.
        for batch in np.array_split(rng.permutation(len(images)),(len(images)+BATCH-1)//BATCH):
            x=input_image(torch.from_numpy(images[batch]),device)
            y=torch.as_tensor(marks[batch],device=device,dtype=torch.float32)/255.
            weight=torch.as_tensor(weights[batch],device=device,dtype=torch.float32)*(1.+4.*y)
            raw=model(x)
            loss=(F.binary_cross_entropy_with_logits(raw,y,reduction="none")*weight).sum()/weight.sum().clamp_min(1.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(batch)
        losses.append(total/len(images))
    return model.eval(),dict(images=len(images),loss=losses)


@torch.no_grad()
def infer(model,images,device):
    shape=images.shape
    flat=images.reshape(-1,*shape[2:])
    chunks=[]
    for start in range(0,len(flat),BATCH):
        x=input_image(torch.from_numpy(flat[start:start+BATCH]),device)
        chunks.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(chunks).reshape(shape[:2]+(SIZE[1],SIZE[0]))


def probability(heat,index,grid,lut):
    frame,station,side=index
    u,v,_=lut[SIDES[side]]
    k=np.rint((abs(grid)-DISTANCES[0])/.2).astype(int)
    u=u[station,k]
    if side:
        u=319.-u
    x=((u+.5)/2.-.5).astype("f")[None,:]
    y=((v[station,k]-CROP[0]+.5)/2.-.5).astype("f")[None,:]
    return cv2.remap(heat[frame,side],x,y,cv2.INTER_LINEAR)[0]


def pick(mode,heat,index,grid,x,lut):
    visual=x[:,0]
    if mode=="contrast":
        return grid[np.argmax(visual)]
    p=probability(heat,index,grid,lut)
    score=p if mode=="heat" else p+.15*(visual-visual.max())/max(np.ptp(visual),1.)
    return grid[np.argmax(score)]


def metrics_dev(group,heat,lut):
    truth,eligible=group["truth"],group["eligible"]
    rows=[(i,g,x) for i,g,x in group["options"] if eligible[i]]
    result={}
    for mode in MODES:
        delta=np.array([abs(pick(mode,heat,i,g,x,lut))-abs(truth[i]) for i,g,x in rows])
        result[mode]=dict(points=len(delta),mae_m=float(abs(delta).mean()),
                          within_1_25_m=int((abs(delta)<=1.25).sum()),signed_m=float(delta.mean()))
    return result


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources={Path(__file__),old.MOUNT,Path(candidates.__code__.co_filename),Path(pixel_targets.__code__.co_filename),
             Path(labels.__code__.co_filename),Path(baseline_data.__code__.co_filename)}
    lut=sampling(old.read(old.MOUNT))
    dev={}
    for track in DEV:
        dev[track]=arrays(track,sources)
        print("Prepared",track,len(dev[track]["x"]),flush=True)
    folds={}
    for held,group in dev.items():
        model,trace=fit([g for t,g in dev.items() if t!=held],device)
        folds[held]=metrics_dev(group,infer(model,group["images"],device),lut)
        print("Fold",held,trace,folds[held],flush=True)
    def risk(mode):
        r=[folds[t][mode]["mae_m"]/folds[t]["contrast"]["mae_m"] for t in DEV]
        return max(r),sum(r)
    winner=min(MODES,key=risk)
    experiment=min((m for m in MODES if m!="contrast"),key=risk)
    model,trace=fit(list(dev.values()),device)
    checkpoint=OUT/"side-heat.pt"
    torch.save(model.state_dict(),checkpoint)
    frozen=OUT/"training.json"
    assert not any(TEST in str(p) or VAL in str(p) for p in sources)
    old.write(frozen,dict(status="frozen_before_replay",development=DEV,excluded=[TEST,VAL],seed=SEED,
              source_revision=old.revision(),device=str(device),epochs=EPOCHS,folds=folds,choice=winner,
              experimental=experiment,fit=trace,checkpoint_sha256=old.digest(checkpoint),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="dense_side_heatmap_training"))
    print("Frozen",winner,experiment,flush=True)
    observers,settings,pins=load_observer()
    route=old.read(ROUTING/"straight-verification.json")
    output={}
    for track,name,mapping in (
            (TEST,"silverstone",CAPTURE),
            (VAL,"vallelunga",OUT.parent/"vision-vallelunga-local-corner-capture-20261112-01"/"archive-map.json")):
        directory=RAW_ARCHIVE/name
        raw_path=directory/"sensors.npz"
        receipt=old.read(directory/"result.json")
        assert old.digest(raw_path)==receipt["inputs_sha256"]
        with np.load(raw_path,allow_pickle=False) as data:
            raw={key:data[key] for key in data.files}
        keys=list(zip(raw["run"],raw["frame"],strict=True))
        bundles=receipt["archive_bundles"]
        images,available=crops(track,mapping,bundles,keys)
        options=list(candidates(track,mapping,bundles,keys,raw["sections"],raw["motion"]))
        assert all(available[i[0]] for i,_,_ in options)
        heat=infer(model,images,device)
        ref_path=ROUTING/f"{track}-state-distance.jsonl"
        reference=frames(ref_path,route["tracks"][track]["rows_sha256"])
        methods={}
        for mode in ("contrast",experiment):
            sections=raw["sections"].copy()
            for i,grid,x in options:
                assert not np.isfinite(sections[i])
                sections[i]=pick(mode,heat,i,grid,x,lut)
            width=sections[:,:,1]-sections[:,:,0]
            invalid=np.isfinite(width)&((width<=3.)|(width>=20.))
            sections[invalid]=raw["sections"][invalid]
            np.testing.assert_array_equal(sections[np.isfinite(raw["sections"])],raw["sections"][np.isfinite(raw["sections"])])
            sensor=OUT/f"{track}-{mode}-sensors.npz"
            np.savez_compressed(sensor,**dict(raw,sections=sections))
            with np.load(sensor,allow_pickle=False) as data:
                rows,aligned,_=observe(data,observers,settings)
            timeline=OUT/f"{track}-{mode}-frames.jsonl"
            old.jsonl(timeline,rows)
            summary,hits=metrics(rows,reference)
            result=dict(metrics=summary,right_hits=hits,frames_sha256=old.digest(timeline),sensors_sha256=old.digest(sensor))
            if mode=="contrast":
                previous=old.read(CHAINS/"result.json")
                assert result["frames_sha256"]==previous["tracks"][track]["methods"]["baseline"]["frames_sha256"]
            if track==TEST:
                with np.load(directory/"predictions/boundaries.npz",allow_pickle=False) as base:
                    result["mesh"]=mesh_score(aligned,base["estimated"],reference,keys)
            methods[mode]=result
        paths={*pins,raw_path,ref_path,mapping,directory/"result.json",frozen,checkpoint}
        if track==TEST:
            paths.update((MESH,MESH.with_suffix(".json"),directory/"predictions/boundaries.npz"))
        output[track]=dict(methods=methods,source_sha256={str(p):old.digest(p) for p in sorted(paths)})
        print(track,{m:(v["right_hits"],v.get("mesh",{}).get("new_mae_m")) for m,v in methods.items()},flush=True)
    old.write(OUT/"result.json",dict(status="scored",training_sha256=old.digest(frozen),tracks=output,
              termination="dense_side_heatmap_replay"))


if __name__=="__main__":
    main()
