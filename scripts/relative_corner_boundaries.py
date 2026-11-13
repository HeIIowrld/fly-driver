"""Relative-colour boundary contrast used by the alert-recovery candidate."""
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np

from corner_fusion_boundaries import (advance, decide, fuse, observed_sections, read_motion,
                                      section_values, transform, whole_shape_min)
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from train_vision_teacher_line import read

class SurfaceContrast(PixelRoadBoundary):
    """Compare same-view strips either side of a candidate surface transition."""

    def __init__(self, mount, *, near=False):
        super().__init__(mount)
        self.stations = np.array([2.,4.,6.,8.]) if near else np.arange(8., 26., 2.)
        self.close = near
        self.lookup = self._sampling(mount, self.stations)

    def detect(self, images):
        sections = []
        center = 0.
        projected = []
        for name, (u, v, valid) in self.lookup.items():
            if not self.close and name != "front":
                continue
            image = cv2.cvtColor(images[name], cv2.COLOR_BGR2LAB)
            height, width = image.shape[:2]
            samples = image[np.rint(np.clip(v, 0, 1)*(height-1)).astype(int),
                            np.rint(np.clip(u, 0, 1)*(width-1)).astype(int)].astype(float)
            projected.append((name, samples, valid))
        candidates = np.arange(9, len(self.lateral)-9)
        for station, distance in enumerate(self.stations):
            offset = np.arange(1, 5) if self.close else np.arange(2, 9)
            boundaries = []
            for side in (-1, 1):
                inside = candidates[:, None]-side*offset
                outside = candidates[:, None]+side*offset
                scores = np.full(len(candidates), -np.inf)
                for name, samples, valid in projected:
                    row = samples[station]
                    inner, outer = row[inside], row[outside]
                    a, b = np.median(inner, axis=1), np.median(outer, axis=1)
                    difference = b-a
                    chroma = np.linalg.norm(difference[:, 1:], axis=1)
                    inner_chroma = np.linalg.norm(a[:, 1:]-128., axis=1)
                    outer_chroma = np.linalg.norm(b[:, 1:]-128., axis=1)
                    support = valid[station, inside].all(1) & valid[station, outside].all(1)
                    eligible = support & (outer_chroma-inner_chroma >= 6)
                    eligible &= (chroma >= 9) | ((chroma >= 6) & (abs(difference[:, 0]) >= 16))
                    eligible &= (self.lateral[candidates]-center)*side >= 1.3
                    score = chroma + .25*abs(difference[:, 0])
                    scores = np.maximum(scores, np.where(eligible, score, -np.inf))
                if np.isfinite(scores).any():
                    # Select the closest transition supported at three nearby lateral samples.
                    supported = np.convolve(np.isfinite(scores).astype(int), np.ones(3), mode="same") == 3
                    indices = candidates[supported]
                    edge = float(self.lateral[indices[0 if side > 0 else -1]]) if len(indices) else None
                else:
                    edge = None
                boundaries.append(edge)
            right, left = boundaries
            valid = right is not None and left is not None and 3. < left-right < 20.
            if valid and distance > 4.:
                center = .5*(left+right)
            sections.append(dict(forward_x_m=float(distance), right_y_m=right, left_y_m=left, valid=valid))
        return sections


def read_boundaries(archives, bundles, track, config, pins):
    """Yield every capture's boundary candidate and contiguous local travel."""
    mount = read(config)
    near, far = SurfaceContrast(mount, near=True), SurfaceContrast(mount)
    cv2.setNumThreads(1)
    age = pins["geometry_history_s"]
    cut = pins["curvature_threshold_per_m"]
    for run, path in read(archives).items():
        history = deque()
        previous = None
        with CameraArchive(path) as archive:
            if archive.bundle_sha256 != bundles[run]:
                raise ValueError(f"Boundary source bundle changed: {run}")
            for capture in map(json.loads, archive.read("stream.jsonl").splitlines()):
                frame = capture["capture_id"]
                meta = archive.json(Path(capture["file"]).name)
                generation = capture["generation"]
                if meta["track_id"] != track or meta["index"] != frame or meta["generation"] != generation:
                    raise ValueError(f"Boundary capture identity differs: {run}/{frame}")
                views = {v["view_id"]: v for v in capture["views"]}
                usable = (len(capture["views"]) == 4 and set(views) == {"front", "rear", "left", "right"}
                          and not any(r.get(f) for r in (capture, meta) for f in ("ideal_line_shown", "aid_probe_only")))
                motion = read_motion(meta) if usable else None
                step = None
                if motion is not None and previous is not None:
                    prev_frame, prev_generation, prev_motion = previous
                    if frame == prev_frame+1 and generation == prev_generation:
                        step = transform(motion, prev_motion)
                points = observed_sections(archive, views, near, far) if motion is not None else None
                if step is None or points is None:
                    history.clear()
                else:
                    rotation, move = step
                    history = deque((stamp, advance(points, rotation, move))
                                    for stamp, points in history if motion[0]-stamp <= age)
                choice = None
                if points is not None:
                    history.append((motion[0], points))
                    aligned = np.stack([section_values(points) for _, points in history])
                    stamps = np.array([stamp for stamp, _ in history])
                    fused = fuse(aligned, stamps, motion[0], age)
                    choice = decide(whole_shape_min(fused, aligned), cut)
                previous = (frame, generation, motion) if points is not None else None
                yield (run, frame), choice, float(step[1][0]) if step is not None else None
