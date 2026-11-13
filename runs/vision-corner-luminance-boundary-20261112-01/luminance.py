"""Replay luminance-supported road edges against the frozen colour detector."""
import json
from pathlib import Path

import cv2
import numpy as np

from audit_road_edge_projection import section_points
from baseline import PROTOCOL,SILVER,fit_boundaries,publish,summarize_approaches
from corner_fusion_boundaries import stations
from evaluate_fused_corner import frame_key,load_frozen
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from heads import review
from profile_replay import TRACK_ROOT
from recover import baseline_data
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent
FIELDS=("right_y_m","left_y_m")
MODES=("original","luminance","luminance_supported","unsigned_chroma_supported")
MIN_BRIGHTNESS=18.


class EdgeEvidence(SurfaceContrast):
    """Evaluate same-camera strips with alternative contrast signs."""
    def detect_evidence(self, lab, original, mode):
        projected=[]
        for name,(u,v,valid) in self.lookup.items():
            if not self.close and name!="front":
                continue
            image=lab[name]
            height,width=image.shape[:2]
            points=image[np.rint(np.clip(v,0,1)*(height-1)).astype(int),
                         np.rint(np.clip(u,0,1)*(width-1)).astype(int)].astype(float)
            projected.append((points,valid))
        candidates=np.arange(9,len(self.lateral)-9)
        offset=np.arange(1,5) if self.close else np.arange(2,9)
        result=np.full((len(original),2),np.nan)
        center=0.
        for i,row in enumerate(original):
            pair=[row[key] for key in FIELDS]
            for side_idx,side in enumerate((-1,1)):
                if pair[side_idx] is not None:
                    continue
                inside=candidates[:,None]-side*offset
                outside=candidates[:,None]+side*offset
                best=np.full(len(candidates),-np.inf)
                for pixels,valid in projected:
                    inner,outer=pixels[i,inside],pixels[i,outside]
                    a,b=np.median(inner,axis=1),np.median(outer,axis=1)
                    light=np.abs(b[:,0]-a[:,0])
                    chroma=np.linalg.norm(b[:,1:]-a[:,1:],axis=1)
                    inner_spread=np.median(np.abs(inner[:,:,0]-a[:,0,None]),axis=1)
                    mask=valid[i,inside].all(axis=1)&valid[i,outside].all(axis=1)
                    mask &= ((self.lateral[candidates]-center)*side>=1.3)&(inner_spread<=8.)
                    if mode=="unsigned_chroma_supported":
                        eligible=(chroma>=12.) | ((chroma>=8.) & (light>=16.))
                    else:
                        eligible=light>=MIN_BRIGHTNESS
                    strength=light+.6*chroma
                    best=np.maximum(best,np.where(mask&eligible,strength,-np.inf))
                allowed=np.convolve(np.isfinite(best).astype(int),np.ones(3),mode="same")==3
                if mode.endswith("supported"):
                    neighbors=[original[j][FIELDS[side_idx]] for j in (i-1,i+1)
                               if 0<=j<len(original) and original[j][FIELDS[side_idx]] is not None]
                    if neighbors:
                        delta=np.min(np.abs(self.lateral[candidates,None]-neighbors),axis=1)
                        allowed &= delta<=1.25
                    else:
                        allowed[:]=False
                if allowed.any():
                    choice=np.flatnonzero(allowed)[0 if side>0 else -1]
                    pair[side_idx]=float(self.lateral[candidates[choice]])
            if None not in pair and 3.<pair[1]-pair[0]<20.:
                result[i]=pair
                if row["forward_x_m"]>4.:
                    center=.5*sum(pair)
        return result


def map_contrast(geometry,meta,lab,raw):
    views={v["view_id"]:v for v in meta["views"]}
    body=max(views.values(),key=lambda view:view["render_call_finished_clock"])["source_body_after_render"]
    xyz=section_points(geometry,body,stations)[:,::-1]
    look=np.array(body["renderer_only_look"])[[2,0]]
    look/=np.linalg.norm(look)
    normal=np.array([look[0],0.,-look[1]])
    offsets=np.array([-.75,-.45,-.25,.25,.45,.75])
    measurements=[]
    for i,distance in enumerate(stations):
        for j,side in enumerate((-1,1)):
            if not np.isfinite(xyz[i,j]).all():
                continue
            # Ground-truth offsets select pixels for diagnosis only.
            points=xyz[i,j]+side*offsets[:,None]*normal
            for camera in (views if distance<=8. else ("front",)):
                uv,_,inside=project_reference_points(points,views[camera])
                if not inside.all():
                    continue
                image=lab[camera]
                h,w=image.shape[:2]
                xy=np.rint(uv*np.array([w-1,h-1])).astype(int)
                pixels=image[xy[:,1],xy[:,0]].astype(float)
                a,b=np.median(pixels[:3],axis=0),np.median(pixels[3:],axis=0)
                measurements.append(dict(distance_m=float(distance),edge=FIELDS[j],camera=camera,
                    existing=raw[i][FIELDS[j]] is not None,light_delta=float(b[0]-a[0]),
                    chroma_delta=float(np.linalg.norm(b[1:]-a[1:])),
                    outside_chroma_change=float(np.linalg.norm(b[1:]-128)-np.linalg.norm(a[1:]-128))))
    return measurements


def run_track(track, mount, hashes):
    group,baseline,mapping,bundles=baseline_data(track,hashes)
    near,far=SurfaceContrast(mount,near=True),SurfaceContrast(mount)
    near_alt,far_alt=EdgeEvidence(mount,near=True),EdgeEvidence(mount)
    by_key={frame_key(r):r for r in group["rows"]}
    outputs={name:{} for name in MODES}
    evidence=[]
    geometry=None
    if track==SILVER:
        path=TRACK_ROOT/track/"ai/fast_lane.ai"
        assert digest(path)==read(PROTOCOL)["hashes"][str(path.resolve())]
        geometry,_=load_edges(path)
        hashes.add(path)
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
                valid=set(views)=={"front","left","right","rear"} and not any(
                    x.get(flag) for x in (item,meta) for flag in ("ideal_line_shown","aid_probe_only"))
                if valid:
                    images={side:cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name),np.uint8),
                                               cv2.IMREAD_COLOR) for side,v in views.items()}
                    assert all(im is not None for im in images.values())
                    lab={side:cv2.cvtColor(im,cv2.COLOR_BGR2LAB) for side,im in images.items()}
                    close=near.detect(images)
                    far_raw=far.detect({"front":images["front"]})
                    raw=close+far_raw[1:]
                    current=np.array([[r[k] for k in FIELDS] if r["valid"] else [np.nan,np.nan] for r in raw])
                    arrays={mode:np.r_[near_alt.detect_evidence(lab,close,mode),
                                        far_alt.detect_evidence(lab,far_raw,mode)[1:]] for mode in MODES[1:]}
                else:
                    current=np.full((len(stations),2),np.nan)
                    arrays={mode:current for mode in MODES[1:]}
                if track==SILVER:
                    assert np.array_equal(current,baseline[key],equal_nan=True)
                arrays["original"]=current
                known=np.isfinite(current).all(axis=1)
                details=dict(run=run,frame=key[1],truth=row["truth"],label=row["label"],modes={})
                for mode,edges in arrays.items():
                    assert np.array_equal(edges[known],current[known])
                    observable,proposal=fit_boundaries(edges,.002)["hinge"]
                    scored=publish(row,proposal,observable)
                    outputs[mode][key]=scored
                    details["modes"][mode]=dict(added=stations[np.isfinite(edges).all(axis=1)&~known].tolist(),
                        observable=observable,state=scored["state"],direction_hit=scored["direction_hit"],
                        joint_hit=scored["joint_hit"],distance_m=scored["predicted_distance_m"])
                if geometry is not None and valid and row["truth"]=="approach":
                    details["projected_map_contrast"]=map_contrast(geometry,meta,lab,raw)
                if row["truth"]=="approach" or any(v["added"] for v in details["modes"].values()):
                    evidence.append(details)
    summary={}
    for mode,index in outputs.items():
        assert set(index)==set(by_key)
        rows=[index[frame_key(r)] for r in group["rows"]]
        summary[mode]=dict(score=review(group,rows),approach=summarize_approaches(rows))
    path=OUT/f"{track}-frames.jsonl"
    jsonl(path,evidence)
    return dict(summary=summary,rows_sha256=digest(path))


def main():
    cv2.setNumThreads(1)
    mount_path=Path(load_frozen(PROTOCOL.with_name("candidate.json"))["camera_config"])
    hashes={Path(__file__),mount_path,Path(SurfaceContrast.detect.__code__.co_filename),
            Path(fit_boundaries.__code__.co_filename),Path(baseline_data.__code__.co_filename)}
    result={}
    for track in (SILVER,"spa"):
        result[track]=run_track(track,read(mount_path),hashes)
        write(OUT/"result.json",dict(status="evaluated",seed=None,tracks=result,
              source_sha256={str(p):digest(p) for p in sorted(hashes)}))
        print(track,{k:v["score"]["metrics"]["candidate"] for k,v in result[track]["summary"].items()},flush=True)

if __name__=="__main__":
    main()
