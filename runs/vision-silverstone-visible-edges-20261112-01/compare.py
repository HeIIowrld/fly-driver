"""Compare raw-image markings with the independently projected track edge."""
import json
from pathlib import Path

import cv2
import numpy as np

from audit_road_edge_projection import section_points
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from train_vision_teacher_line import digest, read, write

OUT=Path(__file__).parent
CAP=Path("runs/vision-silverstone1967-independent-capture-20261112-05")
LOCK=Path("runs/vision-silverstone1967-independent-protocol-20261111-01/base.json")
STATIONS=np.arange(2.,25.,2.)


def main():
    label=OUT/"manual-right-boundary.json"
    inventory=OUT/"inventory.json"
    annotations=read(label)["frames"]
    proof=read(inventory)
    assert set(annotations)=={str(r["frame"]) for r in proof["records"]}
    for record in proof["records"]:
        assert digest(OUT/f'front-{record["frame"]}.png')==record["png_sha256"]
    geometry=next(Path(p) for p in read(LOCK)["hashes"] if Path(p).name=="fast_lane.ai")
    assert digest(geometry)==read(LOCK)["hashes"][str(geometry.resolve())]
    edges,_=load_edges(geometry)
    rows={}
    with CameraArchive(read(CAP/"archive-map.json")["ks_silverstone1967"]) as archive:
        assert archive.bundle_sha256==read(CAP/"result.json")["archive"]["bundle_sha256"]
        for r in map(json.loads,archive.read("stream.jsonl").splitlines()):
            frame=str(r["capture_id"])
            if frame not in annotations:
                continue
            name=Path(r["file"]).name
            assert archive.by_name[name]["sha256"]==r["file_sha256"]
            meta=archive.json(name)
            assert meta["index"]==r["capture_id"] and meta["track_id"]=="ks_silverstone1967"
            front=next(v for v in meta["views"] if v["view_id"]=="front")
            xyz=section_points(edges,front["source_body_after_render"],STATIONS)[:,1]
            valid=np.isfinite(xyz).all(axis=1)
            uv,_,visible=project_reference_points(xyz[valid],front)
            points=uv[visible]*[319,179]
            manual=np.array(annotations[frame]["right_px"])
            records=[]
            for distance,(x,y) in zip(STATIONS[valid][visible],points,strict=True):
                signed=None
                perpendicular=None
                if len(manual) and manual[0,1]<=y<=manual[-1,1]:
                    signed=float(x-np.interp(y,manual[:,1],manual[:,0]))
                    start=manual[:-1]
                    segment=np.diff(manual,axis=0)
                    delta=np.array([x,y])-start
                    t=np.clip(np.sum(delta*segment,axis=1)/np.sum(segment*segment,axis=1),0.,1.)
                    perpendicular=float(np.linalg.norm(delta-t[:,None]*segment,axis=1).min())
                records.append(dict(distance_m=float(distance),map_px=[float(x),float(y)],
                                    map_minus_manual_x_px=signed,manual_polyline_distance_px=perpendicular))
            image=cv2.imdecode(np.frombuffer((OUT/f"front-{frame}.png").read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            canvas=cv2.resize(image,(960,540),interpolation=cv2.INTER_NEAREST)
            if len(manual):
                cv2.polylines(canvas,[np.rint(manual*3).astype(np.int32)],False,(255,225,40),2,cv2.LINE_AA)
            for r in records:
                xy=tuple(np.rint(np.array(r["map_px"])*3).astype(int))
                cv2.circle(canvas,xy,4,(0,195,255),-1,cv2.LINE_AA)
                cv2.putText(canvas,str(int(r["distance_m"])),(xy[0]+4,xy[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX,.39,(0,195,255),1,cv2.LINE_AA)
            cv2.putText(canvas,"Cyan: manually marked transition   Yellow: map edge", (8,23),
                        cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
            ok,encoded=cv2.imencode(".png",canvas)
            assert ok
            (OUT/f"comparison-{frame}.png").write_bytes(encoded.tobytes())
            rows[frame]=dict(kind=annotations[frame]["kind"],projected=records)
    assert set(rows)==set(annotations)
    source=(Path(__file__),label,inventory,geometry,LOCK,CAP/"result.json",CAP/"archive-map.json")
    summary={}
    for kind in {r["kind"] for r in rows.values()}:
        values=[p["map_minus_manual_x_px"] for r in rows.values() if r["kind"]==kind
                for p in r["projected"] if p["map_minus_manual_x_px"] is not None]
        nearest=[p["manual_polyline_distance_px"] for r in rows.values() if r["kind"]==kind
                 for p in r["projected"] if p["manual_polyline_distance_px"] is not None]
        summary[kind]=dict(median_polyline_distance_px=float(np.median(nearest)) if nearest else None,frames=sum(r["kind"]==kind for r in rows.values()),comparable_points=len(values),
            median_abs_delta_px=float(np.median(np.abs(values))) if values else None,
            max_abs_delta_px=float(np.max(np.abs(values))) if values else None)
    write(OUT/"result.json",dict(status="evaluated",summary=summary,frames=rows,
          source_sha256={str(p):digest(p) for p in source}))
    print(summary)

if __name__=="__main__":
    main()
