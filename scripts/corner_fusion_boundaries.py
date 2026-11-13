"""Causal boundary readout for the frozen corner-fusion evaluation."""

from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from train_vision_teacher_line import read

stations = np.arange(2., 25., 2.)
radius = 6.



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
                    eligible = support & (inner_chroma <= 13) & (outer_chroma-inner_chroma >= 6)
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


def advance(points, rotation, move):
    return (points-move) @ rotation


def transform(motion, old_motion):
    t, _, _ = motion
    before, speed, yaw = old_motion
    dt = t-before
    if not 0.045 < dt <= .18:
        return None
    theta = yaw*dt
    c, s = np.cos(theta), np.sin(theta)
    move = speed*dt*np.array([np.cos(theta/2), np.sin(theta/2)])
    return np.array([[c, -s], [s, c]]), move


def section_values(points):
    """Only interpolate between observed points separated by at most two sections."""
    result = np.full((len(stations), 2), np.nan)
    for side in range(2):
        curve = points[:, side]
        if len(curve) == 0:
            continue
        order = np.argsort(curve[:, 0])
        x, y = curve[order].T
        stop = np.searchsorted(x, stations)
        for i, j in enumerate(stop):
            if j < len(x) and abs(x[j]-stations[i]) < 1e-8:
                result[i, side] = y[j]
            elif 0 < j < len(x) and x[j]-x[j-1] <= 4.1:
                fraction = (stations[i]-x[j-1])/(x[j]-x[j-1])
                result[i, side] = y[j-1]+fraction*(y[j]-y[j-1])
    return result


def fuse(values, times, now, age):
    stack = values[(now-times) <= age]
    available = np.isfinite(stack).all(axis=-1)
    fused = np.full((len(stations), 2), np.nan)
    for i, valid in enumerate(available.T):
        # A fused section needs two separately captured views of the same location.
        if valid.sum() >= 2:
            fused[i] = np.median(stack[valid, i], axis=0)
    return fused


def curvature(sections):
    samples = []
    usable = np.isfinite(sections).all(axis=1)
    width = sections[:, 1]-sections[:, 0]
    usable &= (width > 3.) & (width < 20.)
    for center in np.arange(4., 22.):
        offset = stations-center
        mask = usable & (abs(offset) <= radius)
        x = offset[mask]
        if len(x) < 4 or x[0] >= 0 or x[-1] <= 0 or max(np.diff(x)) > 4.:
            continue
        basis = np.column_stack((np.ones(len(x)), x, x*x))
        coefficients = np.linalg.lstsq(basis, sections[mask], rcond=None)[0]
        k = 2.*coefficients[2]/(1.+coefficients[1]**2)**1.5
        signed = float(np.copysign(min(abs(k)), k[0])) if k[0]*k[1] > 0 else 0.
        samples.append((float(center), signed))
    return samples


def decide(samples, cut):
    if len(samples) < 2 or not any(b[0]-a[0] == 1. for a, b in zip(samples, samples[1:])):
        return None
    for (x, a), (y, b) in zip(samples, samples[1:]):
        if y-x == 1. and a*b > 0 and min(abs(a), abs(b)) >= cut:
            return (1 if a > 0 else 2, x)
    return 0, None



def read_motion(meta):
    view = max(meta["views"], key=lambda v: v["render_call_finished_clock"], default={})
    body = view.get("source_body_after_render")
    if not body or not body.get("physics_available") or body.get("local_angular_velocity") is None:
        return None
    values = [body.get("source_clock_finished"), body.get("speed_m_s"), body["local_angular_velocity"][1]]
    if all(v is not None and np.isfinite(v) for v in values) and values[1] >= 0:
        return values
    return None


def observed_sections(archive, views, near, far):
    images = {}
    for name, view in views.items():
        data = archive.read(Path(view["file"]).name)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        images[name] = image
    sections = near.detect(images) + far.detect({"front": images["front"]})[1:]
    pairs = np.array([[s["forward_x_m"], s["right_y_m"], s["left_y_m"]]
                      for s in sections if s["valid"]]).reshape(-1, 3)
    return np.stack((np.repeat(pairs[:, :1], 2, axis=1), pairs[:, 1:]), axis=-1)


def whole_shape_min(fused, aligned):
    raw = curvature(fused)
    samples = [dict(curvature(shape)) for shape in aligned]
    lower = []
    for station, signed in raw:
        direction = np.sign(signed)
        observed = [direction * sample[station] for sample in samples if station in sample]
        bound = min(observed) if len(observed) >= 2 else 0.
        lower.append((station, float(direction * min(abs(signed), max(bound, 0.)))))
    return lower


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
