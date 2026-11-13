"""Project Magione map edges into archived camera pixels for label QA."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_csp_geometry import camera_geometry
from flybrain_drive.diagnostic_path_projection import project_reference_points
from train_next_corner import BASE, MAGIONE
from train_vision_teacher_line import CameraArchive, VIEWS, check, digest, jsonl, read, write


DISTANCES = np.array([10., 15., 20.])


def road_edges(rows):
    center = np.array([r["native_xyz"] for r in rows])
    tangent = (np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0))[:, [2, 0]]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack((tangent[:, 0], np.zeros(len(center)), -tangent[:, 1]))
    return [center + sign * np.array([r[f"{side}_m"] for r in rows])[:, None] * normal
            for side, sign in (("left", 1), ("right", -1))]


def section_points(edges, body, distances=DISTANCES):
    origin = np.asarray(body["renderer_only_position"])
    forward = np.asarray(body["renderer_only_look"])[[2, 0]]
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    points = np.full((len(distances), 2, 3), np.nan)
    for side, edge in enumerate(edges):
        offset = (edge - origin)[:, [2, 0]]
        x, y = offset @ forward, offset @ left
        next_x = np.roll(x, -1)
        end = np.roll(edge, -1, axis=0)
        for i, distance in enumerate(distances):
            hits = np.flatnonzero(((x <= distance) & (distance < next_x))
                                 | ((next_x <= distance) & (distance < x)))
            ratio = (distance - x[hits]) / (next_x[hits] - x[hits])
            lateral = y[hits] + ratio * (np.roll(y, -1)[hits] - y[hits])
            local = np.flatnonzero(np.abs(lateral) < 16.)
            if len(local) == 1:
                k = local[0]
                j = hits[k]
                points[i, side] = edge[j] + ratio[k] * (end[j] - edge[j])
    width = (points[:, 0] - points[:, 1])[:, [2, 0]] @ left
    points[~((width >= 1.8) & (width <= 24.))] = np.nan
    return points


def audit_episode(directory, captures, edges, output):
    records, preview = [], []
    sample = set(np.array(sorted(captures))[np.linspace(0, len(captures)-1, 4, dtype=int)])
    with CameraArchive(directory) as archive:
        stream = [json.loads(line) for line in archive.read("stream.jsonl").splitlines()]
        for row in stream:
            frame = row["capture_id"]
            if frame not in captures:
                continue
            check(not row["ideal_line_shown"] and not row["aid_probe_only"], "camera aid")
            meta = archive.json(Path(row["file"]).name)
            check(meta["index"] == frame and meta["track_id"] == "magione", "camera identity")
            camera_geometry(meta["views"])
            views = {v["view_id"]: v for v in meta["views"]}
            anchor = max(views.values(), key=lambda v: v["render_call_finished_clock"])
            points = section_points(edges, anchor["source_body_after_render"]).reshape(-1, 3)
            valid = np.isfinite(points).all(axis=1)
            coverage = np.zeros(len(points), dtype=bool)
            projections, images = {}, []
            for name in VIEWS:
                view = views[name]
                uv, _, inside = project_reference_points(points[valid], view)
                indices = np.flatnonzero(valid)[inside]
                pixels = uv[inside] * [view["width"]-1, view["height"]-1]
                coverage[indices] = True
                projections[name] = [{"distance_m": float(DISTANCES[j//2]), "edge": ("left", "right")[j%2],
                                      "xy_px": xy.tolist()} for j, xy in zip(indices, pixels)]
                if frame in sample:
                    image = cv2.imdecode(np.frombuffer(archive.read(view["file"]), np.uint8), cv2.IMREAD_COLOR)
                    check(image is not None and image.shape[:2] == (view["height"], view["width"]), "image shape")
                    for j, xy in zip(indices, pixels):
                        p = tuple(np.rint(xy).astype(int))
                        color = (0, 225, 255) if j % 2 == 0 else (255, 200, 0)
                        cv2.circle(image, p, 3, color, -1)
                        cv2.putText(image, str(int(DISTANCES[j//2])), (p[0]+4,p[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, .34, color, 1)
                    cv2.putText(image, f"{frame} {name}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                    images.append(image)
            records.append(dict(frame=frame, geometry=valid.reshape(-1,2).all(1).tolist(),
                                in_image=coverage.reshape(-1,2).all(1).tolist(), pixels=projections))
            if images:
                preview.append(np.hstack(images))
        check(len(records) == len(captures), "missing requested frames")
        bundle = archive.bundle_sha256
    jsonl(output.with_suffix(".jsonl"), records)
    check(cv2.imwrite(str(output.with_suffix(".png")), np.vstack(preview)), "contact sheet write")
    return dict(bundle_sha256=bundle, n=len(records),
                geometry_n=np.sum([r["geometry"] for r in records], axis=0).tolist(),
                in_image_n=np.sum([r["in_image"] for r in records], axis=0).tolist(),
                sheet_sha256=digest(output.with_suffix(".png")), labels_sha256=digest(output.with_suffix(".jsonl")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    geometry = BASE / "heldout-magione-geometry-01/geometry.json"
    try:
        check(digest(geometry) == read(geometry.parent/"result.json")["geometry_sha256"], "geometry receipt")
        edges = road_edges(read(geometry)["rows"])
        rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines()]
        results = {}
        for episode, sub in MAGIONE:
            captures = {r["frame"] for r in rows if r["run"] == episode and r["split"] == "external"}
            results[episode] = audit_episode(BASE/episode/sub/"vision", captures, edges, args.out/episode)
        write(args.out/"result.json", dict(revision=args.revision, program_sha256=digest(Path(__file__)),
            geometry_sha256=digest(geometry), rows_sha256=digest(args.rows), seed=None, results=results,
            actor_input=False, depth_observed=False, termination="projected labels and contact sheets exported"))
    except Exception as error:
        write(args.out/"failure.json", dict(termination=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    main()
