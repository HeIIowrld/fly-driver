"""Check image tracing geometry and quantify the replay against visible transitions."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from image_trace import FrontTrace, at_rows, trace

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
STATIONS = np.arange(2., 35., 2.)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def checks(tracer):
    evidence = np.zeros((180, 320), dtype=float)
    for row in range(90, 130):
        column = 230 + (row - 110)
        evidence[row, column - 1:column + 2] = [12, 28, 12]
    evidence[112] = 0
    curve = trace(evidence, (249, 129), 90)
    np.testing.assert_allclose(curve[[95, 111, 113, 126]], [215., 231., 233., 246.])
    assert np.isnan(curve[112]) and np.isnan(at_rows(curve, [112.5])[0])
    known_lateral = np.array([-5., 5.])
    curves = np.empty((180, 2))
    rows = np.arange(180, dtype=float)
    # At fixed lateral, the level pinhole gives x as an affine function of y.
    for j, lateral in enumerate(known_lateral):
        slope = (159.5 - tracer.scale[3] * lateral - 159.5) / (tracer.rows[3] - 89.5)
        curves[:, j] = 159.5 + slope * (rows - 89.5)
    projected = tracer.project(curves)
    np.testing.assert_allclose(projected[3:], np.broadcast_to(known_lateral, projected[3:].shape), atol=1e-9)


def distances(points, polyline):
    before, after = polyline[:-1], polyline[1:]
    delta = after - before
    offset = points[:, None, :] - before[None, :, :]
    scale = np.clip((offset * delta).sum(axis=-1) / (delta * delta).sum(axis=-1), 0., 1.)
    return np.linalg.norm(offset - scale[:, :, None] * delta, axis=-1).min(axis=1)


def image_metrics(diagnostic, labels, baseline, tracer):
    result = {}
    for index, side in enumerate(("right", "left")):
        curve = np.full(180, np.nan)
        for y, x in diagnostic["trace"][index]:
            curve[y] = x
        mark = np.asarray(labels[side], dtype=float)
        same_points, baseline_points, traced_points = [], [], []
        for item in baseline:
            point = item["selected_px"]
            if item["side"] != side or item["station_m"] < 26. or point is None:
                continue
            if not mark[:, 0].min() <= point[0] <= mark[:, 0].max():
                continue
            baseline_points.append(point)
            i = int(item["station_m"] / 2.) - 1
            y = tracer.rows[i]
            x = at_rows(curve, [y])[0]
            if np.isfinite(x) and mark[:, 0].min() <= x <= mark[:, 0].max():
                traced_points.append([x, y])
                same_points.append(point)
        mask = np.isfinite(curve)
        curve_points = np.c_[curve[mask], np.flatnonzero(mask)]
        curve_points = curve_points[(curve_points[:, 0] >= mark[:, 0].min()) &
                                    (curve_points[:, 0] <= mark[:, 0].max()) &
                                    (curve_points[:, 1] >= mark[:, 1].min() - 1.) &
                                    (curve_points[:, 1] <= mark[:, 1].max() + 1.)]
        result[side] = dict(
            base_far_points=len(baseline_points),
            paired_base_px=distances(np.asarray(same_points), mark).tolist() if same_points else [],
            paired_trace_px=distances(np.asarray(traced_points), mark).tolist() if traced_points else [],
            trace_points_in_label_extent=len(curve_points),
            visible_label_to_trace_px=distances(mark, curve_points).tolist()
            if len(curve_points) >= 2 and np.max(np.diff(curve_points[:, 1])) <= 1. else [],
        )
    return result


def main():
    root = RUNS.parent
    mount = read(root / "configs/ac-csp-exterior-front-v3-retina.json")
    tracer = FrontTrace(mount["views"]["front"], STATIONS, (180, 320))
    checks(tracer)
    receipt = read(OUT / "result.json")
    assert len(receipt["tracks"]) == 2
    sources = dict(receipt["source_sha256"])
    for track, result in receipt["tracks"].items():
        assert track not in result["training"]
        timeline = OUT / f"{track}-frames.jsonl"
        assert digest(timeline) == result["rows_sha256"]
        rows = [json.loads(line) for line in timeline.read_text(encoding="utf8").splitlines()]
        assert len(rows) == len({(r["run"], r["frame"]) for r in rows}) == 1200
        sources.update(result["source_sha256"])
    for name, sha in sources.items():
        assert digest(root / name) == sha, name

    old = RUNS / "vision-corner-far-pixel-trace-20261112-01"
    labels, pixels = old / "visible-transitions.json", old / "pixel-trace.json"
    diagnostics = receipt["tracks"]["ks_silverstone1967"]["diagnostic"]
    assert len(diagnostics) == 13
    marks, baseline = read(labels)["frames"], read(pixels)["frames"]
    image = {frame: image_metrics(diagnostics[frame], info["labels"],
                                  baseline[frame]["rows"], tracer) for frame, info in marks.items()}
    errors, coverage = {}, {}
    for name in ("base", "traced_flat", "traced_map_rows"):
        errors[name] = {side: [value for r in diagnostics.values() for value in r["errors"][name][i]]
                        for i, side in enumerate(("right", "left"))}
        coverage[name] = np.sum([r["points"][name] for r in diagnostics.values()], axis=0).tolist()
    report = dict(status="verified", sources_checked=len(sources), image_diagnostic=image,
                  metric_diagnostic=dict(errors=errors, coverage_right_left=coverage),
                  result_sha256=digest(OUT / "result.json"), labels_sha256=digest(labels),
                  pixel_baseline_sha256=digest(pixels), verification_sha256=digest(Path(__file__)))
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("Two complete timelines, source hashes, synthetic curve and plane inversion verified.")


if __name__ == "__main__":
    main()
