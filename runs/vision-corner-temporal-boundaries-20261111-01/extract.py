"""Extract observed road cross sections and causal local body motion."""
import ast
import json
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
runs = out.parent
reference = runs/"vision-magione-smoothed-sustained-onset-20261111-01"/"audit.py"
config = Path("configs/ac-csp-exterior-front-v3-retina.json")
# This historical script also runs its archive evaluation at import time.
tree = ast.parse(reference.read_text(encoding="utf-8-sig"))
surface = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SurfaceContrast")
exec(compile(ast.Module(body=[surface], type_ignores=[]), str(reference), "exec"))
near_detector = SurfaceContrast(read(config), near=True)
far_detector = SurfaceContrast(read(config))
cv2.setNumThreads(1)

magione = runs/"vision-corner-independent-runner-rehearsal-20261111-01"/"replay"
imola = runs/"vision-corner-imola-full-feature-replay-20261111-01"
inputs = [reference, config, Path(__file__)]
archive_maps = {
    "magione": runs/"vision-frozen-next-corner-20261110-01"/"archives.json",
    "imola": imola/"archives.json",
}
scored = {"magione": magione/"parent"/"score"/"rows.jsonl",
          "imola": runs/"vision-corner-imola-full-parent-20261111-01"/"rows.jsonl"}
expected = read(magione/"result.json")["camera_bundle_sha256_by_run"]
inputs.append(magione/"result.json")
counts, bundles = {}, {}


def decode(archive, view):
    image = cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable recorded camera image")
    return image


def extract():
    for track, archive_map in archive_maps.items():
        inputs.extend((archive_map, scored[track]))
        wanted = {}
        for row in map(json.loads, scored[track].read_text().splitlines()):
            wanted.setdefault(row["run"], set()).add(row["frame"])
        archive_paths = read(archive_map)
        assert wanted.keys() == archive_paths.keys()
        for run, folder in archive_paths.items():
            frames = set()
            with CameraArchive(folder) as archive:
                if track == "magione":
                    assert archive.bundle_sha256 == expected[run]
                bundle = Path(folder)/"archive.json"
                bundles[run] = dict(bundle_sha256=archive.bundle_sha256, manifest_sha256=digest(bundle))
                for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                    frame = item["capture_id"]
                    if frame not in wanted[run]:
                        continue
                    assert frame not in frames
                    frames.add(frame)
                    entry = dict(run=run, frame=frame, generation=item["generation"],
                                 motion=None, sections=None)
                    views = {v["view_id"]: v for v in item["views"]}
                    if set(views) != {"front", "rear", "left", "right"} or item.get("ideal_line_shown") or item.get("aid_probe_only"):
                        yield entry
                        continue
                    meta = archive.json(Path(item["file"]).name)
                    assert meta["index"] == frame and meta["track_id"] == track
                    body = max(meta["views"], key=lambda v: v["render_call_finished_clock"]).get("source_body_after_render")
                    if body and body.get("physics_available") and body.get("local_angular_velocity") is not None:
                        values = [body.get("source_clock_finished"), body.get("speed_m_s"),
                                  body["local_angular_velocity"][1]]
                        if all(v is not None and np.isfinite(v) for v in values) and values[1] >= 0:
                            entry["motion"] = values
                    images = {name: decode(archive, v) for name, v in views.items()}
                    sections = near_detector.detect(images) + far_detector.detect({"front": images["front"]})[1:]
                    entry["sections"] = [[s["forward_x_m"], s["right_y_m"], s["left_y_m"]]
                                         for s in sections if s["valid"]]
                    yield entry
            assert frames == wanted[run]
            counts[run] = len(frames)
            print(track, run, len(frames), flush=True)


rows = out/"observations.jsonl"
jsonl(rows, extract())
write(out/"extraction.json", dict(frames=counts, archive_hashes=bundles,
      source_hashes={str(p): digest(p) for p in inputs}, observations_sha256=digest(rows),
      revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda", seed=None, termination="archive_end",
      actor_inputs="camera pixels and causal local angular rate, speed, capture timing and generation"))
