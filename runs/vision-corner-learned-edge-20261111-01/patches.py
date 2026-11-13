"""Build visual edge-patch features and separate projected AI-boundary labels."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from connected import VIEWS, stations
from corner_fusion_boundaries import read_motion
from evaluate_fused_corner import frame_key, load_frozen, load_score
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from profile_replay import TRACKS, TRACK_ROOT, archive_path, sources
from reference import reference_sections
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent
GRID = np.arange(-12., 12.01, .5)
OFFSETS = np.array([-1., -.5, 0., .5, 1.])
SIDES = np.array([-1., 1.])
CHANNELS = len(OFFSETS) * 3


class PatchGrid(PixelRoadBoundary):
    def __init__(self, mount):
        self.lateral = np.arange(-13., 13.01, .5)
        self.views = tuple(mount["views"])
        patch_x = GRID[None, :, None] + SIDES[:, None, None] * OFFSETS[None, None, :]
        indices = np.rint((patch_x - self.lateral[0]) / .5).astype(int)
        projection = self._sampling(mount, stations)
        self.lookup = {}
        rank = []
        for view in self.views:
            u, v, valid = projection[view]
            coords = np.stack((u[:, indices], v[:, indices]), axis=-1)
            okay = valid[:, indices].all(axis=-1)
            if view != "front":
                okay[stations >= 10.] = False
            centrality = np.abs(coords[:, :, :, 2] - .5).sum(axis=-1)
            rank.append(np.where(okay, centrality, np.inf))
            self.lookup[view] = coords
        rank = np.stack(rank)
        self.preferred = rank.argmin(axis=0)
        self.valid = np.isfinite(rank.min(axis=0))

    def features(self, images):
        values = np.zeros((*self.valid.shape, CHANNELS), dtype=np.uint8)
        sizes = {}
        for i, view in enumerate(self.views):
            image = images[view]
            h, w = image.shape[:2]
            sizes[view] = np.array([w - 1, h - 1])
            pixels = np.rint(self.lookup[view].clip(0., 1.) * sizes[view]).astype(int)
            selected = self.valid & (self.preferred == i)
            u, v = pixels[selected].transpose(2, 0, 1)
            values[selected] = image[v, u].reshape(-1, CHANNELS)
        return values, sizes

    def labels(self, points, metadata, sizes):
        target = np.full(self.valid.shape, -1, dtype=np.int8)
        for j, view in enumerate(self.views):
            for side in range(2):
                xyz = points[:, side]
                valid = np.isfinite(xyz).all(axis=1)
                uv, _, visible = project_reference_points(xyz[valid], metadata[view])
                for k, pos in zip(np.flatnonzero(valid)[visible], uv[visible], strict=True):
                    selected = self.valid[k, side] & (self.preferred[k, side] == j)
                    if not selected.any():
                        continue
                    indices = np.flatnonzero(selected)
                    patch = self.lookup[view][k, side, indices, 2]
                    delta = np.abs((patch - pos) * sizes[view])
                    closest = int(np.square(delta).sum(axis=1).argmin())
                    if delta[closest, 0] > 5. or delta[closest, 1] > 5.:
                        continue
                    near = (delta[:, 0] <= 5.) & (delta[:, 1] <= 5.)
                    negative = (np.abs(GRID[indices] - GRID[indices[closest]]) >= 1.) & ~near
                    target[k, side, indices[negative]] = 0
                    target[k, side, indices[closest]] = 1
        # Camera changes can nominate two neighboring samples. Keep positives and negatives separate.
        return target


def capture_features(archive, capture, meta, detector, edges):
    views = {v["view_id"]:v for v in capture["views"]}
    shape = detector.valid.shape
    if (len(capture["views"]) != 4 or set(views) != VIEWS or read_motion(meta) is None
            or any(r.get(k) for r in (capture, meta) for k in ("ideal_line_shown", "aid_probe_only"))):
        return np.zeros((*shape, CHANNELS), np.uint8), np.full(shape, -1, np.int8), np.full((len(stations), 2), np.nan), False
    images = {}
    for name, view in views.items():
        pixels = cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name), np.uint8),
                              cv2.IMREAD_COLOR)
        if pixels is None:
            return np.zeros((*shape, CHANNELS), np.uint8), np.full(shape, -1, np.int8), np.full((len(stations), 2), np.nan), False
        images[name] = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)
    values, sizes = detector.features(images)
    render_views = {v["view_id"]:v for v in meta["views"]}
    anchor = max(render_views.values(), key=lambda v:v["render_call_finished_clock"])
    pose = anchor["source_body_after_render"]
    if not pose or not pose.get("physics_available"):
        return values, np.full(shape, -1, np.int8), np.full((len(stations), 2), np.nan), True
    xyz, lateral = reference_sections(edges, pose, stations)
    labels = detector.labels(xyz, render_views, sizes)
    return values, labels, lateral, True


def run(track):
    final, base, protocol_path = sources(track)
    protocol = load_frozen(protocol_path)
    config = Path(protocol["camera_config"])
    mapping = archive_path(track)
    final_path, base_path = final / "result.json", base / "result.json"
    proof = read(final_path)
    base_protocol = Path(protocol["base_protocol"])
    geometry = TRACK_ROOT / track / "ai/fast_lane.ai"
    if (digest(mapping) != read(base_path)["archives_sha256"]
            or digest(protocol_path) != proof["protocol_sha256"]
            or digest(base_path) != proof["base_receipt_sha256"]
            or digest(geometry) != read(base_protocol)["hashes"][str(geometry.resolve())]):
        raise ValueError(f"Changed sources for {track}")
    rows = load_score(final / "score", proof["score_receipt_sha256"])
    keys = [frame_key(r) for r in rows]
    edges, _ = load_edges(geometry)
    detector = PatchGrid(read(config))
    cv2.setNumThreads(1)
    features, targets, reference, usable, seen = [], [], [], [], []
    for tag, path in read(mapping).items():
        with CameraArchive(path) as archive:
            if archive.bundle_sha256 != proof["camera_bundle_sha256_by_run"][tag]:
                raise ValueError(f"Changed camera bundle: {tag}")
            for capture in map(json.loads, archive.read("stream.jsonl").splitlines()):
                filename = Path(capture["file"]).name
                if archive.by_name[filename]["sha256"] != capture["file_sha256"]:
                    raise ValueError("Camera frame hash differs")
                meta = archive.json(filename)
                frame = capture["capture_id"]
                if (meta["track_id"], meta["index"], meta["generation"]) != (track, frame, capture["generation"]):
                    raise ValueError("Camera frame identity differs")
                x, y, ref, valid = capture_features(archive, capture, meta, detector, edges)
                features.append(x)
                targets.append(y)
                reference.append(ref)
                usable.append(valid)
                seen.append((tag, frame))
    if seen != keys:
        raise ValueError("Capture ordering differs")
    output = OUT / f"{track}-patches.npz"
    target = np.stack(targets)
    np.savez_compressed(output, features=np.stack(features), labels=target, reference=np.stack(reference),
                        valid=detector.valid, usable=np.array(usable), run=np.array([k[0] for k in keys]),
                        frame=np.array([k[1] for k in keys]))
    inputs = (Path(__file__), config, protocol_path, base_protocol, geometry, mapping, final_path, base_path,
              Path("src/flybrain_drive/pixel_road_boundary.py"), Path("src/flybrain_drive/diagnostic_path_projection.py"),
              Path("runs/vision-corner-edge-reference-20261111-01/reference.py"), Path("scripts/audit_road_edge_projection.py"))
    write(output.with_suffix(".json"), dict(track=track, frames=len(rows), revision=proof["revision"],
          patches_sha256=digest(output), positives=int((target == 1).sum()), negatives=int((target == 0).sum()),
          input_sha256={str(p):digest(p) for p in inputs}, termination="projected_boundary_patch_extraction"))
    print(track, len(rows), int((target == 1).sum()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="+", choices=TRACKS)
    for name in parser.parse_args().tracks:
        run(name)
