"""Audit installed edge reference strips in the frozen and recorded projections."""
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from audit_road_edge_projection import section_points
from evaluate_fused_corner import frame_key, read_rows
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
runs = Path("runs")
installed = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks")
mount_path = Path("configs/ac-csp-exterior-front-v3-retina.json")
mount = read(mount_path)["views"]
distances = np.arange(2., 25., 2.)
cv2.setNumThreads(1)


def project_body(x, y, camera):
    point = np.column_stack((x, y, np.zeros_like(x)))
    delta = point - camera["origin_body_m"]
    depth = delta @ camera["forward_body"]
    if not (depth > 0).all():
        return None
    u = .5+.5*camera["projection_x"]*(delta@camera["right_body"])/depth
    v = .5-.5*camera["projection_y"]*(delta@camera["up_body"])/depth
    return np.column_stack((u, v))


def strip_features(lab, uv, count):
    if uv is None or not ((uv >= .02) & (uv <= .98)).all():
        return None
    h, w = lab.shape[:2]
    xy = np.rint(uv * [w-1, h-1]).astype(int)
    values = lab[xy[:, 1], xy[:, 0]].astype(float)
    inner, outer = (np.median(x, axis=0) for x in (values[:count], values[count:2*count]))
    inner_chroma = float(np.linalg.norm(inner[1:]-128.))
    outer_chroma = float(np.linalg.norm(outer[1:]-128.))
    chroma_delta = float(np.linalg.norm((outer-inner)[1:]))
    luminance_delta = float(outer[0]-inner[0])
    neutral = inner_chroma <= 13.
    outward = outer_chroma-inner_chroma >= 6.
    contrast = chroma_delta >= 9. or (chroma_delta >= 6. and abs(luminance_delta) >= 16.)
    return dict(inner_chroma=inner_chroma, outward_chroma=outer_chroma-inner_chroma,
                chroma_delta=chroma_delta, luminance_delta=luminance_delta,
                neutral=neutral, outward=outward, contrast=contrast, frozen=neutral and outward and contrast,
                without_neutral=outward and contrast, without_outward=neutral and contrast,
                relative_contrast=contrast,
                luminance_fallback=(neutral and outward and contrast) or abs(luminance_delta) >= 25.)


def inspect(archive, stream, reference):
    meta = archive.json(Path(stream["file"]).name)
    views = {v["view_id"]: v for v in meta["views"]}
    body = max(views.values(), key=lambda v: v["render_call_finished_clock"])["source_body_after_render"]
    origin = np.asarray(body["renderer_only_position"])
    look = np.asarray(body["renderer_only_look"])[[2, 0]]
    left = np.array([look[0], 0., -look[1]])/np.linalg.norm(look)
    labs = {}
    for view in stream["views"]:
        image = cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name), np.uint8), cv2.IMREAD_COLOR)
        labs[view["view_id"]] = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    points = section_points(reference, body, distances)
    for i, (distance, boundaries) in enumerate(zip(distances, points, strict=True)):
        if not np.isfinite(boundaries).all():
            continue
        offsets = (np.arange(1, 5) if distance <= 8. else np.arange(2, 9))*.1
        for side, edge in zip((1., -1.), boundaries, strict=True):
            signed = np.r_[-offsets, offsets, 0.] * side
            exact_points = edge + signed[:, None] * left
            lateral = float((edge-origin)@left)
            for name in labs if distance <= 8. else ("front",):
                exact, _, visible = project_reference_points(exact_points, views[name])
                fixed = project_body(np.full_like(signed, distance), lateral+signed, mount[name])
                lab = labs[name]
                row = dict(station_m=float(distance), edge="left" if side > 0 else "right", view=name,
                           exact=strip_features(lab, exact if visible.all() else None, len(offsets)),
                           fixed=strip_features(lab, fixed, len(offsets)))
                if row["fixed"] is not None and row["exact"] is not None:
                    h, w = lab.shape[:2]
                    row["center_offset_px"] = float(np.linalg.norm((fixed[-1]-exact[-1])*[w-1,h-1]))
                yield row


cases = {
    "ks_zandvoort": (runs/"vision-zandvoort-independent-evaluation-20261111-01"/"score",
                     runs/"vision-zandvoort-independent-capture-20261111-02"/"archive-map.json"),
    "magione": (runs/"vision-corner-warning-end-runner-20261111-01"/"magione"/"score",
                runs/"vision-frozen-next-corner-20261110-01"/"archives.json"),
}
data, inputs = [], [Path(__file__), mount_path]
for track, (score, archive_map) in cases.items():
    proof = read(score/"result.json")
    score_file = score/"rows.jsonl"
    assert digest(score_file) == proof["rows_sha256"]
    wanted = {frame_key(r) for r in read_rows(score_file) if r["truth"] == "approach"}
    geometry = installed/track/"ai/fast_lane.ai"
    edges, _ = load_edges(geometry)
    inputs.extend((geometry, archive_map, score/"result.json", score_file))
    for run, folder in read(archive_map).items():
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == proof["camera_bundle_sha256_by_run"][run]
            for row in map(json.loads, archive.read("stream.jsonl").splitlines()):
                key = run, row["capture_id"]
                if key in wanted:
                    data.extend(dict(track=track, run=run, frame=key[1], **r)
                                for r in inspect(archive, row, edges))
                    wanted.remove(key)
    assert not wanted
    print(track, sum(r["track"] == track for r in data), flush=True)

summary = {}
conditions = ("neutral", "outward", "contrast", "frozen", "without_neutral", "without_outward",
              "relative_contrast", "luminance_fallback")
for track in cases:
    matched = [r for r in data if r["track"] == track and r["fixed"] and r["exact"]]
    summary[track] = dict(matched_strips=len(matched),
                          projection_offset_median_px=float(np.median([r["center_offset_px"] for r in matched])),
                          projection_offset_p90_px=float(np.quantile([r["center_offset_px"] for r in matched], .9)))
    for mode in ("fixed", "exact"):
        strip = [r[mode] for r in matched]
        pairs = {}
        for name in conditions:
            counts = Counter((r["run"],r["frame"],r["station_m"],r["edge"]) for r in matched if r[mode][name])
            pairs[name] = sum((*key[:3],"left") in counts and (*key[:3],"right") in counts
                              for key in counts if key[3] == "left")
        summary[track][mode] = dict(passes={c:sum(x[c] for x in strip) for c in conditions},
             paired_stations=pairs, median_inner_chroma=float(np.median([r["inner_chroma"] for r in strip])),
             median_outward_chroma=float(np.median([r["outward_chroma"] for r in strip])))
path=out/"rows.jsonl"
jsonl(path,data)
write(out/"result.json",dict(summary=summary, scope="spline reference used only for after-the-fact projection audit",
      seed=None, termination="archive_end", rows_sha256=digest(path), inputs={str(p):digest(p) for p in inputs}))
print(summary)
