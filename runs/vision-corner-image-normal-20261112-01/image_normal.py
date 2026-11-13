"""Replay a boundary detector that samples across image-space edge normals."""
import json
from pathlib import Path

import cv2
import numpy as np

from baseline import PROTOCOL,SILVER,fit_boundaries,publish,summarize_approaches
from corner_fusion_boundaries import stations
from evaluate_fused_corner import frame_key,load_frozen
from flybrain_drive.ac_camera_archive import CameraArchive
from heads import review
from recover import baseline_data
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent
FIELDS=("right_y_m","left_y_m")
MODES=("baseline","normal_fill","normal_supported")
OFFSETS=(2.,3.,4.)


def across_normal(lab,u,v,visible):
    """Use the strongest Lab image derivative to orient the cross-edge samples."""
    image=cv2.GaussianBlur(lab.astype(np.float32),(3,3),.7)
    dx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    dy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    height,width=image.shape[:2]
    x=np.asarray(u*(width-1),dtype=np.float32)
    y=np.asarray(v*(height-1),dtype=np.float32)
    gx=cv2.remap(dx,x,y,cv2.INTER_LINEAR)
    gy=cv2.remap(dy,x,y,cv2.INTER_LINEAR)
    channel=(gx*gx+gy*gy).argmax(axis=2,keepdims=True)
    nx=np.take_along_axis(gx,channel,axis=2)[:,:,0]
    ny=np.take_along_axis(gy,channel,axis=2)[:,:,0]
    norm=np.hypot(nx,ny)
    nx/=np.maximum(norm,1.e-6)
    ny/=np.maximum(norm,1.e-6)
    strips=[]
    valid=visible.copy()
    for sign in (-1.,1.):
        samples=[]
        for delta in OFFSETS:
            xx,yy=x+sign*delta*nx,y+sign*delta*ny
            valid &= (xx>=1)&(xx<=width-2)&(yy>=1)&(yy<=height-2)
            samples.append(cv2.remap(image,xx,yy,cv2.INTER_LINEAR))
        strips.append(np.median(samples,axis=0))
    diff=strips[1]-strips[0]
    light=np.abs(diff[:,:,0])
    chroma=np.linalg.norm(diff[:,:,1:],axis=2)
    eligible=valid & (norm>=3.) & ((light>=18.) | (chroma>=13.))
    return np.where(eligible,light+.7*chroma,-np.inf)


def edge_scores(detector,lab):
    scores=[across_normal(lab[name],*lut) for name,lut in detector.lookup.items()
            if detector.close or name=="front"]
    return np.maximum.reduce(scores)


def choose(raw,evidence,grid,mode):
    out=np.full((len(raw),2),np.nan)
    center=0.
    for i,row in enumerate(raw):
        pair=[row[k] for k in FIELDS]
        score=evidence[i]
        for side,(sign,key) in enumerate(zip((-1.,1.),FIELDS,strict=True)):
            if pair[side] is not None or mode=="baseline":
                continue
            eligible=np.isfinite(score)&((grid-center)*sign>=1.3)
            # Two adjacent projected samples must have support.
            eligible &= np.r_[False,np.isfinite(score[:-1])] | np.r_[np.isfinite(score[1:]),False]
            if mode=="normal_supported":
                neighbors=[raw[j][key] for j in (i-1,i+1)
                           if 0<=j<len(raw) and raw[j][key] is not None]
                if not neighbors:
                    continue
                delta=np.min(np.abs(grid[:,None]-neighbors),axis=1)
                eligible &= delta<=1.4
            possible=np.flatnonzero(eligible)
            if len(possible):
                chosen=possible[np.argmax(score[possible])]
                pair[side]=float(grid[chosen])
        if None not in pair and 3.<pair[1]-pair[0]<20.:
            out[i]=pair
            if row["forward_x_m"]>4.:
                center=.5*sum(pair)
    return out


def checks():
    u,v=np.meshgrid(np.linspace(.1,.9,48),np.linspace(.25,.75,51))
    lab=np.full((51,48,3),128,np.uint8)
    lab[:25,:,0]=75
    lab[25:,:,0]=150
    values=across_normal(lab,u,v,np.ones_like(u,dtype=bool))
    assert np.isfinite(values[25]).all()
    assert not np.isfinite(values[0]).any()
    lab[:,:,0]=100
    assert not np.isfinite(across_normal(lab,u,v,np.ones_like(u,dtype=bool))).any()


def run_track(track,mount,hashes):
    group,old,mapping,bundles=baseline_data(track,hashes)
    near,far=SurfaceContrast(mount,near=True),SurfaceContrast(mount)
    by_key={frame_key(r):r for r in group["rows"]}
    outputs={mode:{} for mode in MODES}
    details=[]
    for run,folder in read(mapping).items():
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[run]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key=(run,item["capture_id"])
                if key not in by_key:
                    continue
                row=by_key[key]
                name=Path(item["file"]).name
                assert archive.by_name[name]["sha256"]==item["file_sha256"]
                meta=archive.json(name)
                assert meta["track_id"]==track and meta["index"]==key[1] and meta["generation"]==item["generation"]
                views={v["view_id"]:v for v in item["views"]}
                usable=set(views)=={"front","left","right","rear"} and not any(
                    x.get(k) for x in (item,meta) for k in ("ideal_line_shown","aid_probe_only"))
                if usable:
                    images={side:cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name),np.uint8),
                                              cv2.IMREAD_COLOR) for side,v in views.items()}
                    assert all(img is not None for img in images.values())
                    raw=near.detect(images)+far.detect({"front":images["front"]})[1:]
                    lab={side:cv2.cvtColor(img,cv2.COLOR_BGR2LAB) for side,img in images.items()}
                    evidence=np.r_[edge_scores(near,lab),edge_scores(far,lab)[1:]]
                    arrays={mode:choose(raw,evidence,near.lateral,mode) for mode in MODES}
                else:
                    arrays={mode:np.full((len(stations),2),np.nan) for mode in MODES}
                baseline=arrays["baseline"]
                if track==SILVER:
                    assert np.array_equal(baseline,old[key],equal_nan=True)
                paired=np.isfinite(baseline).all(axis=1)
                record=dict(run=run,frame=key[1],truth=row["truth"],label=row["label"],
                            truth_endpoints_m=row["truth_endpoints_m"],methods={})
                for mode,edges in arrays.items():
                    assert np.array_equal(edges[paired],baseline[paired])
                    available,proposal=fit_boundaries(edges,.002)["hinge"]
                    scored=publish(row,proposal,available)
                    outputs[mode][key]=scored
                    record["methods"][mode]=dict(added=stations[np.isfinite(edges).all(axis=1)&~paired].tolist(),
                        observable=available,state=scored["state"],direction_hit=scored["direction_hit"],
                        joint_hit=scored["joint_hit"],distance_m=scored["predicted_distance_m"])
                if row["truth"]=="approach" or any(x["added"] for x in record["methods"].values()):
                    details.append(record)
    scores={}
    for mode,index in outputs.items():
        assert set(index)==set(by_key)
        rows=[index[frame_key(row)] for row in group["rows"]]
        scores[mode]=dict(score=review(group,rows),approach=summarize_approaches(rows))
    path=OUT/f"{track}-frames.jsonl"
    jsonl(path,details)
    return dict(methods=scores,rows_sha256=digest(path))


def main():
    cv2.setNumThreads(1)
    checks()
    mount=Path(load_frozen(PROTOCOL.with_name("candidate.json"))["camera_config"])
    hashes={Path(__file__),mount,Path(SurfaceContrast.detect.__code__.co_filename),
            Path(baseline_data.__code__.co_filename),Path(fit_boundaries.__code__.co_filename)}
    results={}
    for track in (SILVER,"spa"):
        results[track]=run_track(track,read(mount),hashes)
        print(track,{m:x["score"]["metrics"]["candidate"] for m,x in results[track]["methods"].items()},flush=True)
    write(OUT/"result.json",dict(status="evaluated",seed=None,tracks=results,
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
