"""Offline map-edge substitution to localize failures of the visual onset fit."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from audit_road_edge_projection import section_points
from evaluate_fused_corner import frame_key, load_score
from evaluate_live_corner_geometry import load_edges, load_track_truth
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from hinge import fit
from profile_replay import TRACKS, TRACK_ROOT, archive_path, sources
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
BOUNDARY = Path("runs/vision-corner-boundary-hinge-20261111-01")
EXAMPLES = {974, 1181}


def scalar(value):
    return float(value) if np.isfinite(value) else None


def reference_sections(edges, body, stations):
    xyz = section_points(edges, body, stations)[:, ::-1]  # right, left
    origin = np.asarray(body["renderer_only_position"])
    direction = np.asarray(body["renderer_only_look"])[[2, 0]]
    direction /= np.linalg.norm(direction)
    left = np.array([-direction[1], direction[0]])
    lateral = (xyz - origin)[:, :, [2, 0]] @ left
    return xyz, lateral


def preview(archive, capture, meta, xyz, observed, stations, mount):
    front = next(v for v in meta["views"] if v["view_id"] == "front")
    image = cv2.imdecode(np.frombuffer(archive.read(Path(front["file"]).name), np.uint8),
                         cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    image = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    dimensions = np.array([image.shape[1] - 1, image.shape[0] - 1])
    camera = mount["views"]["front"]
    for side in range(2):
        valid = np.isfinite(xyz[:, side]).all(axis=1)
        uv, _, inside = project_reference_points(xyz[valid, side], front)
        for point in np.rint(uv[inside] * dimensions).astype(int):
            cv2.circle(image, tuple(point), 4, (60, 240, 245), 1, cv2.LINE_AA)

        valid = np.isfinite(observed[:, side])
        points = np.column_stack((stations[valid], observed[valid, side], np.zeros(valid.sum())))
        delta = points - camera["origin_body_m"]
        depth = delta @ camera["forward_body"]
        visible = depth > 0.
        delta, depth = delta[visible], depth[visible]
        u = .5 + .5 * camera["projection_x"] * (delta @ camera["right_body"]) / depth
        v = .5 - .5 * camera["projection_y"] * (delta @ camera["up_body"]) / depth
        for x, y in np.rint(np.column_stack((u, v)) * dimensions).astype(int):
            if 0 <= x < dimensions[0] and 0 <= y < dimensions[1]:
                cv2.drawMarker(image, (x, y), (255, 110, 30), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    cv2.putText(image, "AI reference: yellow circles   Visual samples: blue crosses",
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1, cv2.LINE_AA)
    path = OUT / f"spa-{capture['capture_id']}-front.png"
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError(f"Cannot encode {path}")
    path.write_bytes(data.tobytes())
    return path


def evaluate_capture(row, visual, reference, stations, body, diagnostic):
    valid = np.isfinite(visual).all(axis=1) & np.isfinite(reference).all(axis=1)
    same_mask = np.where(np.isfinite(visual), reference, np.nan)
    variants = {"visual": visual, "reference_same_mask": same_mask, "reference_full": reference}
    fitted = {name: fit(stations, values) for name, values in variants.items()}
    output = dict(run=row["run"], frame=row["frame"], truth=row["truth"], label=row["label"],
                  truth_endpoints_m=row["truth_endpoints_m"], physical_corner=row["physical_corner"],
                  diagnostic_truth=diagnostic, visual_sections=int(np.isfinite(visual).all(axis=1).sum()),
                  ref_sections=int(np.isfinite(reference).all(axis=1).sum()), shared_sections=int(valid.sum()),
                  signed_lateral_error_m=[[scalar(v) for v in pair] for pair in (visual - reference)[valid]],
                  fits=fitted)
    if row["frame"] in EXAMPLES and row["run"] == "spa":
        output.update(stations_m=stations.tolist(),
                      visual=[[scalar(v) for v in pair] for pair in visual],
                      reference=[[scalar(v) for v in pair] for pair in reference],
                      body_look_y=float(body["renderer_only_look"][1]))
    return output


def summarize(rows):
    report = {}
    for category in ("left", "right", "straight", "too_far", "too_near", "in_corner"):
        selected = []
        for row in rows:
            kind = row["truth"]
            if kind == "approach":
                kind = "left" if row["label"] == "1" else "right"
            if kind == "out_of_range":
                kind = "too_far" if np.mean(row["truth_endpoints_m"]) > 20. else "too_near"
            if category == kind:
                selected.append(row)
        absolute = [abs(v) for r in selected for pair in r["signed_lateral_error_m"] for v in pair]
        metrics = dict(frames=len(selected), shared_edge_points=len(absolute),
                       edge_mae_m=float(np.mean(absolute)) if absolute else None,
                       edge_median_absolute_m=float(np.median(absolute)) if absolute else None)
        for mode in ("visual", "reference_same_mask", "reference_full"):
            counts = Counter(observable=0, offers=0, direction=0, joint=0)
            for row in selected:
                possible, offer = row["fits"][mode]
                counts["observable"] += possible
                if offer is None:
                    continue
                counts["offers"] += 1
                if row["truth"] == "approach" and row["label"] == str(offer[1]):
                    counts["direction"] += 1
                    counts["joint"] += max(abs(offer[2] - v) for v in row["truth_endpoints_m"]) <= 5.
            metrics[mode] = counts
        report[category] = metrics
    return report


def run():
    hashes = {Path(__file__), BOUNDARY / "hinge.py", Path("scripts/audit_road_edge_projection.py"),
              Path("scripts/evaluate_live_corner_geometry.py"),
              Path("src/flybrain_drive/diagnostic_path_projection.py")}
    result, all_rows, previews = {}, [], {}
    for track in TRACKS:
        final, base, protocol_path = sources(track)
        protocol = read(protocol_path)
        final_path, base_path = final / "result.json", base / "result.json"
        proof, base_proof = read(final_path), read(base_path)
        rows = load_score(final / "score", proof["score_receipt_sha256"])
        keyed = {frame_key(row): row for row in rows}
        geometry = TRACK_ROOT / track / "ai/fast_lane.ai"
        base_protocol = Path(protocol["base_protocol"])
        config = Path(protocol["camera_config"])
        mapping = archive_path(track)
        pins = read(base_protocol)["hashes"]
        if (pins[str(geometry.resolve())] != digest(geometry)
                or digest(mapping) != base_proof["archives_sha256"]
                or digest(protocol_path) != proof["protocol_sha256"]):
            raise ValueError(f"Changed geometry audit inputs for {track}")
        edges, _ = load_edges(geometry)
        truth, _, _ = load_track_truth(geometry)
        observations = BOUNDARY / f"{track}-sections.npz"
        receipt = observations.with_suffix(".json")
        if digest(observations) != read(receipt)["sections_sha256"]:
            raise ValueError(f"Visual edge arrays changed: {track}")
        with np.load(observations) as cached:
            if list(zip(cached["run"], cached["frame"], strict=True)) != list(keyed):
                raise ValueError(f"Visual edge alignment differs: {track}")
            stations = cached["stations"]
            visual = dict(zip(keyed, cached["values"][:, 0], strict=True))
        hashes.update((final_path, base_path, geometry, base_protocol, protocol_path, config,
                       mapping, observations, receipt))
        track_rows = []
        for tag, path in read(mapping).items():
            with CameraArchive(path) as archive:
                if archive.bundle_sha256 != proof["camera_bundle_sha256_by_run"][tag]:
                    raise ValueError(f"Bundle differs: {tag}")
                for capture in map(json.loads, archive.read("stream.jsonl").splitlines()):
                    key = tag, capture["capture_id"]
                    row = keyed[key]
                    if row["truth"] not in ("approach", "straight", "out_of_range", "in_corner"):
                        continue
                    name = Path(capture["file"]).name
                    if archive.by_name[name]["sha256"] != capture["file_sha256"]:
                        raise ValueError(f"Frame receipt differs: {key}")
                    meta = archive.json(name)
                    if (meta["track_id"], meta["index"], meta["generation"]) != (track, key[1], capture["generation"]):
                        raise ValueError(f"Frame identity differs: {key}")
                    last = max(meta["views"], key=lambda v:v["render_call_finished_clock"])
                    body = last.get("source_body_after_render")
                    if body is None or not body.get("physics_available"):
                        raise ValueError(f"Registered capture missing diagnostic pose: {key}")
                    xyz, reference = reference_sections(edges, body, stations)
                    output = evaluate_capture(row, visual[key], reference, stations, body, truth(body))
                    track_rows.append(dict(track=track, **output))
                    if track == "spa" and key[1] in EXAMPLES:
                        image = preview(archive, capture, meta, xyz, visual[key], stations, read(config))
                        previews[image.name] = digest(image)
        all_rows.extend(track_rows)
        result[track] = summarize(track_rows)
        print(track, len(track_rows), result[track]["left"], flush=True)
    if len(previews) != len(EXAMPLES):
        raise ValueError("Missing diagnostic frames")
    detail = OUT / "rows.jsonl"
    jsonl(detail, all_rows)
    write(OUT / "result.json", dict(tracks=result, diagnostic_rows=len(all_rows),
          rows_sha256=digest(detail), image_sha256=previews,
          inputs_sha256={str(p):digest(p) for p in sorted(hashes)},
          seed=None, termination="privileged_map_edge_substitution_complete"))


if __name__ == "__main__":
    run()
