"""Verify the paired replay and compare its traces with independent image marks."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from annotate import OUT, digest

OLD = OUT.parent / "vision-corner-image-row-trace-20261112-01" / "supported"
ROOT = OUT.parents[1]
TOLERANCE = 2.
MARGIN = .75
DARK = (620, 621, 622, 623)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def distance(points, segments):
    if not len(segments):
        return np.full(len(points), np.inf)
    start, end = segments[:, 0], segments[:, 1]
    step = end - start
    offset = points[:, None, :] - start
    t = np.clip((offset * step).sum(axis=-1) / (step * step).sum(axis=1), 0., 1.)
    return np.linalg.norm(offset - t[:, :, None] * step, axis=-1).min(axis=1)


def segments(points):
    return np.stack((points[:-1], points[1:]), axis=1)


def analyze_frame(diagnostic, annotation):
    target = np.asarray(annotation["target"], dtype=float)
    competitor = annotation.get("competitor")
    points = np.asarray(diagnostic["trace"][0], dtype=float).reshape(-1, 2)[:, ::-1]
    # Use a common band that includes both edges and preserves off-line points.
    in_scope = points[(points[:, 0] >= target[0, 0]) & (points[:, 0] <= target[-1, 0])
                      & (points[:, 1] >= 85.) & (points[:, 1] <= 116.)]
    near_target = distance(in_scope, segments(target))
    if competitor is not None:
        near_other = distance(in_scope, segments(np.asarray(competitor, dtype=float)))
        categories = np.full(len(in_scope), "other", dtype="<U20")
        categories[(near_target <= TOLERANCE) | (near_other <= TOLERANCE)] = "overlap"
        categories[(near_target <= TOLERANCE) & (near_other - near_target >= MARGIN)] = "road_facing"
        categories[(near_other <= TOLERANCE) & (near_target - near_other >= MARGIN)] = "far_side"
    else:
        categories = np.where(near_target <= TOLERANCE, "road_facing", "other")
    connected = segments(points)[np.diff(points[:, 1]) == 1.]
    mark_errors = distance(target, connected)
    marks = [float(error) if np.isfinite(error) else None for error in mark_errors]
    return dict(trace_points=len(in_scope), classification=dict(Counter(categories.tolist())),
                annotated_point_errors_px=marks, annotated_points_hit=int((mark_errors <= TOLERANCE).sum()))


def render(labels, original, signed):
    canvas = Image.new("RGB", (1000, 4 * 145), "#171717")
    draw = ImageDraw.Draw(canvas)
    origin_x, origin_y = 200, 85
    scale = 2
    box = (origin_x, origin_y, 320, 134)
    for i, frame in enumerate(DARK):
        key = str(frame)
        entry = labels[key]
        image = Image.open(ROOT / entry["source_path"]).convert("RGB")
        crop = image.crop(box).resize(((box[2]-box[0])*scale, (box[3]-box[1])*scale),
                                      Image.Resampling.NEAREST)
        for j, (label, result) in enumerate((("Native pixels", None), ("Visible transition labels", None),
                                              ("Absolute luminance", original), ("Signed luminance", signed))):
            x, y = 10+j*249, 24+i*145
            canvas.paste(crop, (x, y))
            draw.text((x, y-17), f"{frame}  {label}", fill="white")

            def mapped(points):
                return [(round(x+(px-origin_x)*scale), round(y+(py-origin_y)*scale))
                        for px, py in points]

            if j == 1:
                draw.line(mapped(entry["target"]), fill="#50E3AD", width=2)
                draw.line(mapped(entry["competitor"]), fill="#FFB85E", width=2)
            if result is not None:
                points = [(px, py) for py, px in result[key]["trace"][0]
                          if origin_x <= px < box[2] and origin_y <= py < box[3]]
                for (px, py), (qx, qy) in zip(points, points[1:]):
                    if qy - py == 1:
                        draw.line(mapped([(px, py), (qx, qy)]), fill="#A8BEFF", width=2)
                for px, py in mapped(points):
                    draw.point((px, py), fill="#FFFFFF")
    canvas.save(OUT / "dark-strip-comparison.png")


def metric_diagnostic(result, track):
    rows = result["tracks"][track]["diagnostic"].values()
    errors = [value for row in rows for value in row["errors"]["traced_flat"][0]]
    return dict(individually_projected_right_sections=sum(row["points"]["traced_flat"][0] for row in rows),
                reference_overlap_points=len(errors), lateral_median_m=float(np.median(errors)) if errors else None)


def main():
    labels_path = OUT / "labels.json"
    labels = read(labels_path)["frames"]
    assert len(labels) == 13 and sum(a["target"] is not None for a in labels.values()) == 7
    for annotation in labels.values():
        assert digest(ROOT / annotation["source_path"]) == annotation["sha256"]
    path = OUT / "result.json"
    baseline_path = OLD / "result.json"
    baseline, signed = read(baseline_path), read(path)
    assert digest(baseline_path) == read(OLD / "verification.json")["result_sha256"]
    for source in (baseline, signed):
        hashes = dict(source["source_sha256"])
        for track, report in source["tracks"].items():
            assert track not in report["training"]
            hashes.update(report["source_sha256"])
        for filename, sha in hashes.items():
            assert digest(ROOT / filename) == sha, filename
    timelines, metrics = {}, {}
    for track in signed["tracks"]:
        records = []
        for run, result in ((OLD, baseline), (OUT, signed)):
            timeline = run / f"{track}-frames.jsonl"
            assert digest(timeline) == result["tracks"][track]["rows_sha256"]
            rows = [json.loads(line) for line in timeline.read_text(encoding="utf8").splitlines()]
            assert len(rows) == len({(row["run"], row["frame"]) for row in rows}) == 1200
            records.append(rows)
        for a, b in zip(*records, strict=True):
            for name in ("run", "frame", "truth", "label", "truth_endpoints_m", "physical_corner"):
                assert a[name] == b[name], (track, name, a["frame"])
            for name in ("edge_base", "t4t5_base", "edge_far_wide", "t4t5_far_wide"):
                assert a["decisions"][name] == b["decisions"][name], (track, name, a["frame"])
        differences = []
        for a, b in zip(*records, strict=True):
            if a["truth"] == "approach" and a["label"] == "2" and (
                    a["offers"].get("trace_replace") != b["offers"].get("trace_replace")
                    or a["decisions"]["t4t5_trace_replace"] != b["decisions"]["t4t5_trace_replace"]):
                differences.append(dict(frame=a["frame"], truth_m=a["truth_endpoints_m"],
                                        candidates=[r["offers"].get("trace_replace") for r in (a,b)],
                                        t4t5=[r["decisions"]["t4t5_trace_replace"] for r in (a,b)]))
        timelines[track] = differences
        metrics[track] = {}
        for run, key in zip(records, ("absolute", "signed"), strict=True):
            right = [r for r in run if r["truth"] == "approach" and r["label"] == "2"]
            metrics[track][key] = {mode: sum(r["pairs"].get(mode, 0) for r in right)
                                   for mode in ("far_wide", "trace_fill", "trace_replace")}

    key = "ks_silverstone1967"
    diagnostics = (baseline["tracks"][key]["diagnostic"], signed["tracks"][key]["diagnostic"])
    assert set(labels) == set(diagnostics[0]) == set(diagnostics[1])
    image = {frame: {name: analyze_frame(result[frame], annotation)
                     for name, result in zip(("absolute", "signed"), diagnostics, strict=True)}
             for frame, annotation in labels.items() if annotation["target"] is not None}
    render(labels, *diagnostics)
    report = dict(status="verified", seed=None, source_revision=signed["source_revision"],
                  labels_sha256=digest(labels_path), results_sha256=digest(path),
                  baseline_result_sha256=digest(baseline_path),
                  overlay_sha256=digest(OUT / "dark-strip-comparison.png"),
                  verifier_sha256=digest(Path(__file__)), image_diagnostic=image, right_pairs=metrics,
                  metric_diagnostic={name: metric_diagnostic(result, key)
                                     for name, result in (("absolute", baseline), ("signed", signed))},
                  changed_right_frames=timelines,
                  termination="paired_source_hashes_and_visible_transition_annotations_verified")
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("Visible transition annotations, source hashes and 2,400 paired frames verified.")


if __name__ == "__main__":
    main()
