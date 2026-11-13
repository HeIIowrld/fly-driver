"""Trace far boundary picks to the archived camera pixels and diagnostic map projections."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from audit_road_edge_projection import section_points
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from image_normal import FIELDS, choose, edge_scores
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest, read, write

OUT=Path(__file__).parent
RUNS=OUT.parent
CAP=RUNS/"vision-silverstone1967-independent-capture-20261112-05"
OLD=RUNS/"vision-corner-extended-horizon-20261112-01"
MOUNT=Path("configs/ac-csp-exterior-front-v3-retina.json")
LOCK=RUNS/"vision-silverstone1967-independent-protocol-20261111-01/base.json"
TARGET=(620,621,622,623)
STATIONS=np.arange(2.,35.,2.)
COLORS=((255,150,40),(30,210,245))


def context():
    path=OLD/"replay.py"
    spec=importlib.util.spec_from_file_location("horizon_replay",path)
    entry=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def draw(image, pairs, frame, kind):
    """Render a fixed image crop with station marks and a native-pixel grid."""
    x0,y0,x1,y1=35,76,298,121
    scale=4
    canvas=cv2.resize(image[y0:y1,x0:x1],None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    for x in range(40,298,20):
        px=(x-x0)*scale
        cv2.line(canvas,(px,0),(px,len(canvas)-1),(88,88,88),1)
        cv2.putText(canvas,str(x),(px+2,12),cv2.FONT_HERSHEY_SIMPLEX,.33,(240,240,240),1,cv2.LINE_AA)
    for y in range(80,121,10):
        py=(y-y0)*scale
        cv2.line(canvas,(0,py),(canvas.shape[1]-1,py),(88,88,88),1)
        cv2.putText(canvas,str(y),(1,py+12),cv2.FONT_HERSHEY_SIMPLEX,.33,(240,240,240),1,cv2.LINE_AA)
    for row in pairs:
        if row["station_m"]<24:
            continue
        point=row.get(kind)
        if point is None:
            continue
        x,y=point
        if not (x0<=x<x1 and y0<=y<y1):
            continue
        pixel=(round((x-x0)*scale),round((y-y0)*scale))
        color=COLORS[row["side"]=="left"]
        cv2.circle(canvas,pixel,3,color,-1)
        offset=16+12*(int(row["station_m"]//2)%2)
        dy=-offset if kind=="selected_px" else offset
        cv2.line(canvas,pixel,(pixel[0],pixel[1]+dy),color,1)
        cv2.putText(canvas,str(int(row["station_m"])),(pixel[0]-7,pixel[1]+dy),
                    cv2.FONT_HERSHEY_SIMPLEX,.34,color,1,cv2.LINE_AA)
    path=OUT/f"{frame}-{kind}.png"
    ok,encoded=cv2.imencode(".png",canvas)
    assert ok
    path.write_bytes(encoded.tobytes())
    return digest(path)


def analyze(archive,item,meta,near,far,track,entry):
    views={v["view_id"]:v for v in item["views"]}
    images={k:cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name),np.uint8),cv2.IMREAD_COLOR)
            for k,v in views.items()}
    assert all(im is not None for im in images.values())
    raw=near.detect(images)+far.detect({"front":images["front"]})[1:]
    points=np.asarray([[r[k] if r[k] is not None else np.nan for k in FIELDS] for r in raw])
    lab={name:cv2.cvtColor(im,cv2.COLOR_BGR2LAB) for name,im in images.items()}
    evidence=np.r_[edge_scores(near,lab),edge_scores(far,lab)[1:]]
    evidence[:,[0,-1]]=-np.inf
    filled=choose(raw,evidence,near.lateral,"normal_fill")
    selected=np.where(np.isfinite(filled).all(axis=1)[:,None]&np.isfinite(points).any(axis=1)[:,None],filled,points)
    camera=next(v for v in meta["views"] if v["view_id"]=="front")
    ref=section_points(track,camera["source_body_after_render"],STATIONS)[:,::-1]
    near_lut,far_lut=(d.lookup["front"] for d in (near,far))
    u,v,vis=(np.r_[a,b[1:]] for a,b in zip(near_lut,far_lut,strict=True))
    rows=[]
    for i,x in enumerate(STATIONS):
        for s,side in enumerate(("right","left")):
            val=selected[i,s]
            if np.isfinite(val):
                j=int(np.abs(near.lateral-val).argmin())
                pixel=[float(u[i,j]*319),float(v[i,j]*179)] if vis[i,j] else None
            else:
                pixel=None
            world=ref[i,s]
            if np.isfinite(world).all():
                uv,_,visible=project_reference_points(world[None,:],camera)
                reference=(uv[0]*[319,179]).tolist() if visible[0] else None
            else:
                reference=None
            rows.append(dict(station_m=float(x),side=side,selected_px=pixel,reference_px=reference,
                             source="surface_transition" if np.isfinite(points[i,s]) else
                                    "normal_fill" if np.isfinite(val) else "missing",
                             lateral_m=float(val) if np.isfinite(val) else None))
    frame=item["capture_id"]
    digests={kind:draw(images["front"],rows,frame,kind) for kind in ("selected_px","reference_px","raw")}
    return dict(rows=rows,offers=entry.proposals(selected),images=digests)


def main():
    cv2.setNumThreads(1)
    entry=context()
    pins=read(LOCK)["hashes"]
    track_path=next(Path(p) for p in pins if Path(p).name=="fast_lane.ai")
    assert digest(track_path)==pins[str(track_path.resolve())]
    track,_=load_edges(track_path)
    mount=read(MOUNT)
    near,far=(SurfaceContrast(mount,near=k) for k in (True,False))
    far.stations=STATIONS[3:]
    far.lookup=far._sampling(mount,far.stations)
    mapping,receipt=CAP/"archive-map.json",CAP/"result.json"
    recorded=read(OLD/"result.json")["tracks"]["ks_silverstone1967"]["oracle"]
    output={}
    with CameraArchive(read(mapping)["ks_silverstone1967"]) as archive:
        assert archive.bundle_sha256==read(receipt)["archive"]["bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            f=item["capture_id"]
            if f not in TARGET:
                continue
            name=Path(item["file"]).name
            assert archive.by_name[name]["sha256"]==item["file_sha256"]
            meta=archive.json(name)
            assert (meta["track_id"],meta["index"],meta["generation"])==("ks_silverstone1967",f,item["generation"])
            result=analyze(archive,item,meta,near,far,track,entry)
            assert {k:list(v) if v else None for k,v in result["offers"].items()}==recorded[str(f)]["observed"]
            output[f]=result
    assert set(output)==set(TARGET)
    sources=(Path(__file__),MOUNT,LOCK,track_path,mapping,receipt,OLD/"result.json",OLD/"replay.py",
             Path(project_reference_points.__code__.co_filename),Path(section_points.__code__.co_filename))
    write(OUT/"pixel-trace.json",dict(status="diagnostic_only",frames=output,
          source_sha256={str(p):digest(p) for p in sources},seed=None,termination="offers_reproduced"))
    for r in output[620]["rows"]:
        if r["station_m"]>=26.:
            print(r,flush=True)


if __name__=="__main__":
    main()
