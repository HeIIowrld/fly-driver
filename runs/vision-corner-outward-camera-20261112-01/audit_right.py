"""Audit labelled Silverstone right-edge misses in the original side camera."""
from pathlib import Path
import sys
import json

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from replay_corner_camera_archive import stream, old, image
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from image_normal import across_normal
from verify_directional import frames

TRACK = "ks_silverstone1967"
ARCHIVE = OUT.parent/"vision-corner-archive-observer-20261112-01"/"silverstone"
CAPTURE = OUT.parent/"vision-silverstone1967-independent-capture-20261112-05"/"archive-map.json"
ROUTING = OUT.parent/"vision-corner-recovery-routing-20261112-01"
MESH = OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{TRACK}-train-labels.npz"


def main():
    prior = old.read(ROUTING/"straight-verification.json")
    reference = ROUTING/f"{TRACK}-state-distance.jsonl"
    timeline = frames(reference, prior["tracks"][TRACK]["rows_sha256"])
    sensor = ARCHIVE/"predictions/boundaries.npz"
    assert old.digest(sensor) == old.read(sensor.with_name("result.json"))["boundaries_sha256"]
    assert old.digest(MESH) == old.read(MESH.with_suffix(".json"))["labels_sha256"]
    with np.load(sensor, allow_pickle=False) as data, np.load(MESH, allow_pickle=False) as mesh:
        keys = list(zip(data["run"], data["frame"], strict=True))
        assert keys == list(zip(mesh["run"], mesh["frame"], strict=True)) == [(r["run"], r["frame"]) for r in timeline]
        stations, observed = data["stations"], data["estimated"]
        labelled, pixels, witness = mesh["lateral_m"], mesh["image_px"], mesh["adjacent_invalid"]
        views = mesh["views"].tolist()
    right = views.index("right")
    view_valid = np.isfinite(pixels).all(axis=-1)
    approach = np.array([row["truth"]=="approach" and row["label"]=="2" for row in timeline])
    missing = approach[:, None] & witness[:, :, 0] & view_valid[:, :, :, 0].any(axis=1) & ~np.isfinite(observed[:, :, 0])
    detector = PixelRoadBoundary(old.read(old.MOUNT))
    lookup = detector._sampling(detector.mount, stations)["right"]
    bundle = old.read(ARCHIVE/"result.json")["archive_bundles"][TRACK]
    folder = old.read(CAPTURE)[TRACK]
    sought = {frame: i for i, (_, frame) in enumerate(keys) if missing[i].any()}
    notes, panels = [], []
    for arc, item, meta, available in stream(folder, bundle, TRACK):
        frame = item["capture_id"]
        if frame not in sought:
            continue
        i = sought[frame]
        view = available["right"]
        bitmap = image(arc, view)
        h, w = bitmap.shape[:2]
        score = across_normal(cv2.cvtColor(bitmap, cv2.COLOR_BGR2LAB), *lookup)
        annotated = bitmap.copy()
        for j in np.flatnonzero(missing[i]):
            visible = [name for k, name in enumerate(views) if view_valid[i, k, j, 0]]
            value = float(labelled[i, j, 0])
            p = int(np.argmin(abs(detector.lateral-value)))
            flat = np.array([lookup[0][j, p]*(w-1), lookup[1][j, p]*(h-1)])
            projected = pixels[i, right, j, 0]
            displacement = projected-flat
            near = np.flatnonzero(np.isfinite(score[j]) & (abs(detector.lateral-value)<=1.))
            notes.append(dict(frame=frame, station_m=stations[j], lateral_m=value,
                              other_edge_m=float(observed[i,j,1]) if np.isfinite(observed[i,j,1]) else None,
                              views=visible, physical_pixel=projected.tolist(), flat_pixel=flat.tolist(),
                              dx_dy_px=displacement.tolist(), gradient_support_near_flat=bool(len(near))))
            if np.isfinite(projected).all():
                cv2.circle(annotated, tuple(np.rint(projected).astype(int)), 3, (20, 240, 220), -1)
                cv2.circle(annotated, tuple(np.rint(flat).astype(int)), 4, (20, 40, 250), 1)
        cv2.putText(annotated, str(frame), (10,22), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        panels.append(annotated)
    assert len(notes) == int(missing.sum())
    while len(panels)%3:
        panels.append(np.zeros_like(panels[0]))
    preview = np.vstack([np.hstack(panels[i:i+3]) for i in range(0,len(panels),3)])
    ok, encoded = cv2.imencode(".png", preview)
    assert ok
    (OUT/"right-eye-missing.png").write_bytes(encoded.tobytes())
    result = dict(status="observed_right_boundary_misses_diagnosed", points=notes,
                  only_right_camera=sum(p["views"]==["right"] for p in notes),
                  fixed_projection_mae_px=np.mean(abs(np.array([p["dx_dy_px"] for p in notes])), axis=0).tolist(),
                  gradient_support_near_flat=sum(p["gradient_support_near_flat"] for p in notes),
                  source_sha256={str(p):old.digest(p) for p in (Path(__file__),reference,sensor,MESH,CAPTURE,old.MOUNT)})
    old.write(OUT/"right-camera-audit.json",result)
    print(result["only_right_camera"],result["fixed_projection_mae_px"],result["gradient_support_near_flat"])


if __name__ == "__main__":
    main()
