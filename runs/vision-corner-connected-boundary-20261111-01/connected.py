"""Retain multiple colour transitions and select a connected pair of visual road edges."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from corner_fusion_boundaries import read_motion, stations
from evaluate_fused_corner import frame_key, load_frozen, load_score
from flybrain_drive.ac_camera_archive import CameraArchive
from relative_corner_boundaries import SurfaceContrast
from profile_replay import TRACKS, archive_path, sources
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent
VIEWS = {"front", "rear", "left", "right"}
N_CANDIDATES = 6
SMOOTHNESS = (.25, .60)


class Transitions(SurfaceContrast):
    def hypotheses(self, lab):
        projected = []
        for name, (u, v, valid) in self.lookup.items():
            if not self.close and name != "front":
                continue
            image = lab[name]
            h, w = image.shape[:2]
            samples = image[np.rint(v.clip(0., 1.) * (h - 1)).astype(int),
                            np.rint(u.clip(0., 1.) * (w - 1)).astype(int)].astype(float)
            projected.append((samples, valid))
        idx = np.arange(9, len(self.lateral) - 9)
        offsets = np.arange(1, 5) if self.close else np.arange(2, 9)
        strips = [(idx[:, None] - side * offsets, idx[:, None] + side * offsets)
                  for side in (-1, 1)]
        for section in range(len(self.stations)):
            candidates = []
            for side, (inside, outside) in zip((-1, 1), strips, strict=True):
                scores = np.full(len(idx), -np.inf)
                for samples, visible in projected:
                    row = samples[section]
                    a, b = np.median(row[inside], axis=1), np.median(row[outside], axis=1)
                    diff = b - a
                    chroma = np.linalg.norm(diff[:, 1:], axis=1)
                    inner = np.linalg.norm(a[:, 1:] - 128., axis=1)
                    outer = np.linalg.norm(b[:, 1:] - 128., axis=1)
                    eligible = visible[section, inside].all(1) & visible[section, outside].all(1)
                    eligible &= (outer - inner >= 6) & ((chroma >= 9) | ((chroma >= 6) & (abs(diff[:, 0]) >= 16)))
                    scores = np.maximum(scores, np.where(eligible, chroma + .25 * abs(diff[:, 0]), -np.inf))
                supported = np.convolve(np.isfinite(scores), np.ones(3), mode="same") == 3
                locations = self.lateral[idx]
                if self.stations[section] <= 4.:
                    supported &= side * locations >= 1.3
                candidates.append(retain(locations, np.where(supported, scores, -np.inf), side))
            yield pair_states(*candidates)


def retain(locations, scores, side):
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return np.empty((0, 2))
    # Preserve the inward-most transition and strongest separated alternatives.
    first = finite[0 if side == 1 else -1]
    ranking = [first, *finite[np.argsort(-scores[finite], kind="stable")]]
    keep = []
    for index in ranking:
        if all(abs(locations[index] - locations[k]) >= .8 for k in keep):
            keep.append(index)
            if len(keep) == N_CANDIDATES:
                break
    return np.column_stack((locations[keep], scores[keep]))


def pair_states(right, left):
    r, l = np.meshgrid(np.arange(len(right)), np.arange(len(left)), indexing="ij")
    width = left[l, 0] - right[r, 0]
    ok = (width > 3.) & (width < 20.)
    r, l = r[ok], l[ok]
    quality = .55 + .25 * np.tanh((np.column_stack((right[r, 1], left[l, 1])) - 12.) / 18.)
    return np.column_stack((right[r, 0], left[l, 0], quality.sum(axis=1)))


def connect(station_values, groups, smoothness):
    """Select the most supported path with at most one skipped section per step."""
    states = []
    best = (-np.inf, -1, -1)
    for i, current in enumerate(groups):
        x = station_values[i]
        if not len(current):
            states.append(None)
            continue
        center = current[:, :2].mean(axis=1)
        scores = current[:, 2] - .25 * (x - station_values[0]) - .05 * abs(center)
        links = np.full((len(current), 2), -1, dtype=int)
        for old in range(max(0, i - 2), i):
            if states[old] is None:
                continue
            distance = x - station_values[old]
            if distance > 4.01:
                continue
            previous = groups[old]
            delta = current[:, None, :2] - previous[None, :, :2]
            width_change = abs(delta[:, :, 1] - delta[:, :, 0])
            penalty = smoothness * (np.abs(delta).mean(axis=2) + .75 * width_change) + .65 * (i - old - 1)
            continuity = np.max(np.abs(delta), axis=2) <= .75 + .65 * distance
            continuations = states[old][0][None, :] + current[:, 2, None] - penalty
            continuations = np.where(continuity, continuations, -np.inf)
            predecessor = np.argmax(continuations, axis=1)
            totals = continuations[np.arange(len(current)), predecessor]
            improve = totals > scores
            scores[improve] = totals[improve]
            links[improve, 0], links[improve, 1] = old, predecessor[improve]
        states.append((scores, links))
        j = int(scores.argmax())
        if scores[j] > best[0]:
            best = scores[j], i, j
    out = np.full((len(groups), 2), np.nan)
    _, i, j = best
    while i >= 0:
        out[i] = groups[i][j, :2]
        i, j = states[i][1][j]
    return out


def extract(archive, capture, meta, near, far):
    views = {v["view_id"]: v for v in capture["views"]}
    if (len(capture["views"]) != 4 or set(views) != VIEWS or read_motion(meta) is None
            or any(r.get(k) for r in (capture, meta) for k in ("ideal_line_shown", "aid_probe_only"))):
        return np.full((len(SMOOTHNESS), len(stations), 2), np.nan)
    lab = {}
    for name, view in views.items():
        pixels = cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name), np.uint8),
                              cv2.IMREAD_COLOR)
        if pixels is None:
            return np.full((len(SMOOTHNESS), len(stations), 2), np.nan)
        lab[name] = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)
    groups = [*near.hypotheses(lab), *list(far.hypotheses(lab))[1:]]
    return np.stack([connect(stations, groups, w) for w in SMOOTHNESS])


def run(track):
    final, base, protocol_path = sources(track)
    protocol = load_frozen(protocol_path)
    config = Path(protocol["camera_config"])
    mapping = archive_path(track)
    final_path, base_path = final / "result.json", base / "result.json"
    proof = read(final_path)
    if (digest(mapping) != read(base_path)["archives_sha256"]
            or digest(protocol_path) != proof["protocol_sha256"]
            or digest(base_path) != proof["base_receipt_sha256"]):
        raise ValueError(f"Changed camera inputs: {track}")
    rows = load_score(final / "score", proof["score_receipt_sha256"])
    keys = [frame_key(r) for r in rows]
    mount = read(config)
    near, far = Transitions(mount, near=True), Transitions(mount)
    cv2.setNumThreads(1)
    saved, seen = [], []
    for run_name, path in read(mapping).items():
        with CameraArchive(path) as archive:
            if archive.bundle_sha256 != proof["camera_bundle_sha256_by_run"][run_name]:
                raise ValueError(f"Changed camera bundle: {run_name}")
            for capture in map(json.loads, archive.read("stream.jsonl").splitlines()):
                name = Path(capture["file"]).name
                if archive.by_name[name]["sha256"] != capture["file_sha256"]:
                    raise ValueError("Camera frame hash differs")
                meta = archive.json(name)
                frame = capture["capture_id"]
                if (meta["track_id"], meta["index"], meta["generation"]) != (track, frame, capture["generation"]):
                    raise ValueError("Camera frame identity differs")
                seen.append((run_name, frame))
                saved.append(extract(archive, capture, meta, near, far))
    if keys != seen:
        raise ValueError(f"Camera frame alignment differs: {track}")
    output = OUT / f"{track}-edges.npz"
    np.savez_compressed(output, edges=np.stack(saved), stations=stations,
                        frame=np.array([r["frame"] for r in rows]), run=np.array([r["run"] for r in rows]))
    inputs = (Path(__file__), mapping, config, protocol_path, final_path, base_path,
              Path("scripts/relative_corner_boundaries.py"), Path("scripts/corner_fusion_boundaries.py"))
    write(output.with_suffix(".json"), dict(track=track, frames=len(rows), revision=proof["revision"],
          input_sha256={str(p):digest(p) for p in inputs}, edges_sha256=digest(output),
          smoothness=SMOOTHNESS, max_candidates_per_side=N_CANDIDATES,
          termination="connected_camera_edges_complete"))
    print(track, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="+", choices=TRACKS)
    for track in parser.parse_args().tracks:
        run(track)
