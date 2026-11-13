
"""Identify which physical camera supported a clipped Silverstone section."""

import json
from pathlib import Path
import cv2
import numpy as np

from image_normal import FIELDS, across_normal, choose
from relative_corner_boundaries import SurfaceContrast
from replay import visible_scores
from flybrain_drive.ac_camera_archive import CameraArchive
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent
CAP = OUT.parent / "vision-silverstone1967-independent-capture-20261112-05"
MOUNT = Path("configs/ac-csp-exterior-front-v3-retina.json")
FOCUS = {622: 1, 623: 3}


def main():
    cv2.setNumThreads(1)
    mapping = CAP / "archive-map.json"
    receipt = CAP / "result.json"
    near = SurfaceContrast(read(MOUNT), near=True)
    rows = {}
    with CameraArchive(read(mapping)["ks_silverstone1967"]) as archive:
        assert archive.bundle_sha256 == read(receipt)["archive"]["bundle_sha256"]
        for row in map(json.loads, archive.read("stream.jsonl").splitlines()):
            frame = row["capture_id"]
            if frame not in FOCUS:
                continue
            i = FOCUS[frame]
            images = {v["view_id"]: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                                cv2.IMREAD_COLOR) for v in row["views"]}
            lab = {name: cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for name, im in images.items()}
            raw = near.detect(images)
            scores = {name: across_normal(lab[name], *lut) for name, lut in near.lookup.items()}
            old = np.maximum.reduce(list(scores.values()))
            old[:, [0, -1]] = -np.inf
            original = choose(raw, old, near.lateral, "normal_fill")[i]
            current = choose(raw, visible_scores(near, lab), near.lateral, "normal_fill")[i]
            details = {}
            for side, val in zip(FIELDS, original, strict=True):
                if not np.isfinite(val):
                    continue
                k = int(np.argmin(abs(near.lateral-val)))
                details[side] = dict(lateral_m=float(val), measured=raw[i][side] is not None, cameras={
                    name: dict(score=float(values[i, k]), neighbor_visibility=near.lookup[name][2][i, k-1:k+2].tolist(),
                               uv=[float(l[i, k]) for l in near.lookup[name][:2]])
                    for name, values in scores.items() if np.isfinite(values[i, k])})
            rows[frame] = dict(station_m=raw[i]["forward_x_m"], sides=details,
                               filtered=[float(x) if np.isfinite(x) else None for x in current])
    assert set(rows) == set(FOCUS)
    sources=(Path(__file__), MOUNT, mapping, receipt, Path(visible_scores.__code__.co_filename))
    write(OUT / "visibility-focus.json",dict(rows=rows,source_sha256={str(p):digest(p) for p in sources}))
    print(rows,flush=True)


if __name__ == "__main__":
    main()
