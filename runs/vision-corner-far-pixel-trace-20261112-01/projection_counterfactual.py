"""Intervene on far-view pixel projection for a diagnostic geometry/appearance split."""

import json
from pathlib import Path

import cv2
import numpy as np

from trace import CAP, LOCK, MOUNT, OUT, STATIONS, context
from audit_road_edge_projection import section_points
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from image_normal import FIELDS, choose, edge_scores
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest, read, write


def selections(near, far, images, labs, near_raw, near_scores):
    raw=near_raw+far.detect({"front":images["front"]})[1:]
    evidence=np.r_[near_scores,edge_scores(far,labs)[1:]]
    evidence[:,[0,-1]]=-np.inf
    measured=np.asarray([[r[k] if r[k] is not None else np.nan for k in FIELDS] for r in raw])
    filled=choose(raw,evidence,near.lateral,"normal_fill")
    return np.where(np.isfinite(filled).all(axis=1)[:,None]&np.isfinite(measured).any(axis=1)[:,None],
                    filled,measured)


def displacement(far, track, front, lookup):
    """Compute camera correction only for diagnosis, never from a chosen image edge."""
    body=front["source_body_after_render"]
    xyz=section_points(track,body,far.stations)[:,::-1]
    location=np.asarray(body["renderer_only_position"])
    look=np.asarray(body["renderer_only_look"])[[2,0]]
    normal=np.array([-look[1],look[0]])/np.linalg.norm(look)
    lateral=(xyz-location)[:,:,[2,0]]@normal
    deltas=np.zeros((2,*lookup[0].shape))
    explained=[]
    for i,d in enumerate(far.stations):
        delta=[]
        for s in range(2):
            if not np.isfinite(xyz[i,s]).all():
                continue
            j=np.abs(far.lateral-lateral[i,s]).argmin()
            if not lookup[2][i,j] or abs(far.lateral[j]-lateral[i,s])>.11:
                continue
            pixels,_,visible=project_reference_points(xyz[i,s][None,:],front)
            if not visible[0]:
                continue
            uv=np.array([lookup[0][i,j],lookup[1][i,j]])
            delta.append((lateral[i,s],pixels[0]-uv))
        if len(delta)==2 and delta[0][0]<delta[1][0]:
            for axis in range(2):
                deltas[axis,i]=np.interp(far.lateral,[d[0] for d in delta],[d[1][axis] for d in delta])
            explained.append(dict(station_m=float(d),right_delta_px=(delta[0][1]*[319,179]).tolist(),
                                  left_delta_px=(delta[1][1]*[319,179]).tolist()))
    return deltas,explained


def lookup_with_offset(far,lookup,du,dv):
    u,v,_=lookup
    u,v=u+du,v+dv
    # Re-evaluate image extent while retaining the original front-facing domain.
    forward=np.isfinite(u)&np.isfinite(v)
    far.lookup["front"]=(u,v,forward&(u>=.02)&(u<=.98)&(v>=.02)&(v<=.98))


def main():
    cv2.setNumThreads(1)
    trace=OUT/"pixel-trace.json"
    traced=read(trace)
    entry=context()
    pins=read(LOCK)["hashes"]
    path=next(Path(p) for p in pins if Path(p).name=="fast_lane.ai")
    assert digest(path)==pins[str(path.resolve())]
    track,_=load_edges(path)
    mount=read(MOUNT)
    near,far=(SurfaceContrast(mount,near=k) for k in (True,False))
    far.stations=STATIONS[3:]
    far.lookup=far._sampling(mount,far.stations)
    original=far.lookup["front"]
    mapping,receipt=CAP/"archive-map.json",CAP/"result.json"
    results={}
    with CameraArchive(read(mapping)["ks_silverstone1967"]) as archive:
        assert archive.bundle_sha256==read(receipt)["archive"]["bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            f=str(item["capture_id"])
            if f not in traced["frames"]:
                continue
            filename=Path(item["file"]).name
            assert archive.by_name[filename]["sha256"]==item["file_sha256"]
            meta=archive.json(filename)
            assert (meta["track_id"],meta["index"],meta["generation"])==("ks_silverstone1967",int(f),item["generation"])
            front=next(v for v in meta["views"] if v["view_id"]=="front")
            images={v["view_id"]:cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name),np.uint8),
                                             cv2.IMREAD_COLOR) for v in item["views"]}
            labs={name:cv2.cvtColor(im,cv2.COLOR_BGR2LAB) for name,im in images.items()}
            raw,scores=near.detect(images),edge_scores(near,labs)
            (du,dv),pixels=displacement(far,track,front,original)
            variants={}
            for mode,offsets in {"fixed":(0.,0.),"row":(0.,dv),"column":(du,0.),"both":(du,dv)}.items():
                if mode=="fixed":
                    far.lookup["front"]=original
                else:
                    lookup_with_offset(far,original,*offsets)
                edges=selections(near,far,images,labs,raw,scores)
                proposals=entry.proposals(edges)
                variants[mode]=dict(proposals=proposals,far_pairs=int(np.isfinite(edges[12:]).all(axis=1).sum()))
            assert {k:list(v) if v else None for k,v in variants["fixed"]["proposals"].items()}==traced["frames"][f]["offers"]
            results[f]=dict(pixel_corrections=pixels,variants=variants)
    assert set(results)==set(traced["frames"])
    sources=(Path(__file__),trace,MOUNT,LOCK,path,mapping,receipt,Path(project_reference_points.__code__.co_filename))
    write(OUT/"projection-counterfactual.json",dict(status="diagnostic_only",seed=None,frames=results,
        source_sha256={str(p):digest(p) for p in sources},termination="projection_factors_replayed"))
    print({f:{m:v["proposals"]["far_wide"] for m,v in r["variants"].items()} for f,r in results.items()},flush=True)


if __name__=="__main__":
    main()
