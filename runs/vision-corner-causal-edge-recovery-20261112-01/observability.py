"""Locate lost edge sides in admitted Silverstone approach images."""
import json
from pathlib import Path

import cv2
import numpy as np

from baseline import ARCHIVES, PROTOCOL, SILVER, SILVER_RESULT
from audit_road_edge_projection import section_points
from corner_fusion_boundaries import stations
from evaluate_fused_corner import load_frozen,read_rows
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from profile_replay import TRACK_ROOT
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest,read,write

OUT=Path(__file__).parent

def main():
    rows=SILVER_RESULT/"admission/rows.jsonl"
    wanted={row["frame"]:row for row in read_rows(rows) if row["truth"]=="approach"}
    geometry=TRACK_ROOT/SILVER/"ai/fast_lane.ai"
    assert digest(geometry)==read(PROTOCOL)["hashes"][str(geometry.resolve())]
    edges,_=load_edges(geometry)
    config=Path(load_frozen(PROTOCOL.with_name("candidate.json"))["camera_config"])
    mount=read(config)
    near,far=SurfaceContrast(mount,near=True),SurfaceContrast(mount)
    details=[]
    with CameraArchive(read(ARCHIVES)[SILVER]) as archive:
        assert archive.bundle_sha256==read(ARCHIVES.with_name("result.json"))["archive"]["bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            frame=item["capture_id"]
            if frame not in wanted:
                continue
            name=Path(item["file"]).name
            assert archive.by_name[name]["sha256"]==item["file_sha256"]
            meta=archive.json(name)
            assert meta["track_id"]==SILVER and meta["index"]==frame
            latest=max(meta["views"],key=lambda view:view["render_call_finished_clock"])
            xyz=section_points(edges,latest["source_body_after_render"],stations)[:,::-1]
            camera={v["view_id"]:v for v in meta["views"]}
            images={side:cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name),np.uint8),
                                     cv2.IMREAD_COLOR) for side,view in camera.items()}
            raw=near.detect(images)+far.detect({"front":images["front"]})[1:]
            visibility={}
            for side,view in camera.items():
                points=xyz.reshape(-1,3)
                valid=np.isfinite(points).all(axis=1)
                visible=np.zeros(len(points),dtype=bool)
                _,_,visible[valid]=project_reference_points(points[valid],view)
                visibility[side]=visible.reshape(-1,2).tolist()
            details.append(dict(frame=frame,direction=wanted[frame]["label"],
                corner=wanted[frame]["physical_corner"],raw=raw,map_visibility=visibility))
    assert len(details)==24
    write(OUT/"observation-diagnostic.json",dict(status="evaluated",details=details,
          source_sha256={str(p):digest(p) for p in (Path(__file__),rows,ARCHIVES,geometry,config)}))

if __name__=="__main__":
    main()
