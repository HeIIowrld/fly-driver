"""Inspect boundary input support on the fixed Zandvoort approach frames."""
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from corner_fusion_boundaries import SurfaceContrast, observed_sections, read_motion
from evaluate_fused_corner import frame_key, read_rows
from evaluate_fused_corner_warning_end import capture_motion
from flybrain_drive.ac_camera_archive import CameraArchive
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
receipt = read(out/"result.json")
archive_map = Path("runs/vision-zandvoort-independent-capture-20261111-02/archive-map.json")
score_path = out/"score/rows.jsonl"
assert read(score_path.with_name("result.json"))["rows_sha256"] == digest(score_path)
wanted = {frame_key(r): r for r in read_rows(score_path) if r["truth"] == "approach"}
assert len(wanted) == 58
bundles = receipt["camera_bundle_sha256_by_run"]
travel = dict(capture_motion(archive_map, receipt["track"], bundles))
config = Path("configs/ac-csp-exterior-front-v3-retina.json")
mount = read(config)
near, far = SurfaceContrast(mount, near=True), SurfaceContrast(mount)
cv2.setNumThreads(1)
output = []
for tag, folder in read(archive_map).items():
    with CameraArchive(folder) as archive:
        assert archive.bundle_sha256 == bundles[tag]
        for stream in map(json.loads, archive.read("stream.jsonl").splitlines()):
            key = tag, stream["capture_id"]
            if key not in wanted:
                continue
            meta = archive.json(Path(stream["file"]).name)
            motion = read_motion(meta)
            views = {view["view_id"]:view for view in stream["views"]}
            sections = observed_sections(archive, views, near, far)
            assert sections is not None
            output.append(dict(run=tag, frame=key[1], side=wanted[key]["label"],
                motion_available=motion is not None, local_travel_m=travel[key],
                paired_sections=len(sections), station_m=sections[:, 0, 0].tolist()))
assert {frame_key(r) for r in output} == wanted.keys()
summary = dict(frames=len(output), paired_sections=dict(Counter(r["paired_sections"] for r in output)),
               valid_motion=sum(r["motion_available"] for r in output),
               local_travel_available=sum(r["local_travel_m"] is not None for r in output),
               median_local_travel_m=float(np.median([r["local_travel_m"] for r in output
                                                    if r["local_travel_m"] is not None])))
jsonl(out/"boundary-support.jsonl", output)
write(out/"boundary-support.json", dict(summary=summary, rows_sha256=digest(out/"boundary-support.jsonl"),
      inputs={str(p):digest(p) for p in (Path(__file__), config, archive_map, score_path, out/"result.json")}))
print(summary)
