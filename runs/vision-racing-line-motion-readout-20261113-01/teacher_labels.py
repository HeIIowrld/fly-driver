"""Build future-trajectory teacher targets from verified four-camera recordings.

Renderer positions and heading are consumed here for offline supervision only.
"""
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0, str(RUNS / "vision-corner-fixed-left-20261113-01"))
from fixed_left import DEVELOP, archive_mapping, digests, keys, load_observer, old, read_development
from replay_corner_boundary_states import alt
from train_vision_teacher_line import at, H

MIN_SPEED = 5.
MAX_RELATIVE_STEP_ERROR = 0.35
MIN_STEP_TOLERANCE = 0.65
MAX_LATE_ANCHOR_DELTA_M = 0.8


def read_render_pose(archive, entry, track):
    filename = Path(entry["file"]).name
    if archive.by_name[filename]["sha256"] != entry["file_sha256"]:
        raise ValueError("Camera metadata hash disagrees with its stream")
    meta = archive.json(filename)
    if (meta["track_id"], meta["index"], meta["generation"]) != (track, entry["capture_id"], entry["generation"]):
        raise ValueError("Camera identity differs from the requested track and frame")
    views = meta["views"]
    if len(views) != 4 or {v["view_id"] for v in views} != {"front", "rear", "left", "right"}:
        return None
    if any(item.get(flag) is not False for item in (entry, meta) for flag in ("ideal_line_shown", "aid_probe_only")):
        return None
    first = min(views, key=lambda v: v["render_started_clock"]).get("source_body_before_render")
    last = max(views, key=lambda v: v["render_call_finished_clock"]).get("source_body_after_render")
    if not first or not last or not first["physics_available"] or not last["physics_available"]:
        return None
    fields = ("renderer_only_position", "renderer_only_look", "source_clock_finished", "speed_m_s")
    if any(last.get(k) is None for k in fields) or first.get(fields[0]) is None:
        return None
    early = np.asarray(first[fields[0]], dtype=float)[[2, 0]]
    position = np.asarray(last[fields[0]], dtype=float)[[2, 0]]
    look = np.asarray(last[fields[1]], dtype=float)[[2, 0]]
    time, speed = float(last[fields[2]]), float(last[fields[3]])
    if not np.isfinite([*early, *position, *look, time, speed]).all() or speed < MIN_SPEED:
        return None
    if np.linalg.norm(look) < 0.9 or np.linalg.norm(early-position) > MAX_LATE_ANCHOR_DELTA_M:
        return None
    return time, speed, position, look / np.linalg.norm(look)


def continuous(prev_row, row, prev, pose):
    if not alt.adjacent(prev_row, row):
        return False
    dt = pose[0] - prev[0]
    if not 0.045 < dt <= 0.18:
        return False
    displacement = pose[2] - prev[2]
    distance = np.linalg.norm(displacement)
    expected = dt * (pose[1] + prev[1]) / 2.
    return (distance > 0. and displacement @ prev[3] > 0.5 * distance
            and abs(distance-expected) <= max(MIN_STEP_TOLERANCE, MAX_RELATIVE_STEP_ERROR*expected))


def future_targets(rows, poses, horizons=H):
    targets = np.full((len(rows), len(horizons), 2), np.nan)
    segment = np.full(len(rows), -1, int)
    start = None
    spans = []
    for i, pose in enumerate(poses):
        if start is not None and (pose is None or not continuous(rows[i-1], rows[i], poses[i-1], pose)):
            spans.append((start, i))
            start = None
        if start is None and pose is not None:
            start = i
    if start is not None:
        spans.append((start, len(rows)))
    for group, (start, stop) in enumerate(spans):
        points = np.stack([p[2] for p in poses[start:stop]])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        for j in range(stop-start):
            ahead = arc[j]+horizons
            usable = ahead <= arc[-1]
            if not usable.any():
                continue
            i = start+j
            forward = poses[i][3]
            transform = np.column_stack((forward, [-forward[1], forward[0]]))
            position = (at(arc, points, ahead[usable])-points[j]) @ transform
            # Keep a connected prefix of points ahead of the body.
            n = int(np.logical_and.accumulate(np.diff(np.r_[0.,position[:,0]])>0.).sum())
            targets[i, :n] = position[:n]
            segment[i] = group
    return targets, segment


def build_development():
    _, settings, sources = load_observer()
    mapping, mapping_source = archive_mapping()
    sources.update((Path(__file__), mapping_source, Path(at.__code__.co_filename), Path(alt.adjacent.__code__.co_filename)))
    artifacts = {}
    for track in DEVELOP:
        sensors, expected, proof = read_development(track, sources, settings)
        mapping_path = mapping[track]
        sources.add(mapping_path)
        found = {}
        for run, path in old.read(mapping_path).items():
            with old.CameraArchive(path) as archive:
                if archive.bundle_sha256 != proof["camera_bundle_sha256_by_run"][run]:
                    raise ValueError("Development camera bundle changed")
                for entry in map(json.loads, archive.read("stream.jsonl").splitlines()):
                    key = run, entry["capture_id"]
                    if key in found:
                        raise ValueError("Duplicate frame in the camera bundle")
                    found[key] = read_render_pose(archive, entry, track)
        if set(found) != set(keys(expected)):
            raise ValueError("Development sensor and camera frame inventories differ")
        targets, segment = future_targets(expected, [found[key] for key in keys(expected)])
        path = OUT / f"{track}-development.npz"
        np.savez_compressed(path, **sensors, teacher_xy_m=targets, teacher_segment=segment)
        artifacts[track] = dict(file=str(path), sha256=old.digest(path), frames=len(expected),
                                valid_targets=np.isfinite(targets).all(axis=2).sum(axis=0).tolist(),
                                bundle_sha256=proof["camera_bundle_sha256_by_run"])
        print(track, artifacts[track]["valid_targets"], flush=True)
    old.write(OUT / "development-labels.json", dict(artifacts=artifacts, sources=digests(sources),
              horizons_m=H.tolist(), seed=None, source_revision=old.revision(),
              termination="archive_end", supervision="future renderer trajectory after camera capture"))


if __name__ == "__main__":
    build_development()
