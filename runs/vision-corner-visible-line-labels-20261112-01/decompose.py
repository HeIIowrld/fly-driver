"""Separate tracing, row placement, and reference semantics on matched image points."""
import json
from pathlib import Path

import numpy as np

from annotate import OUT, digest

RUNS = OUT.parent
MOUNT = RUNS.parent / "configs/ac-csp-exterior-front-v3-retina.json"
PIXELS = RUNS / "vision-corner-far-pixel-trace-20261112-01" / "pixel-trace.json"
ABSOLUTE = RUNS / "vision-corner-image-row-trace-20261112-01" / "supported" / "result.json"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sample(xy, row, max_gap=1.01):
    points = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(points) < 2 or not points[0, 1] <= row <= points[-1, 1]:
        return None
    j = np.searchsorted(points[:, 1], row)
    if j < len(points) and abs(points[j, 1] - row) < 1e-10:
        return float(points[j, 0])
    a, b = points[j - 1:j + 1]
    if b[1] - a[1] > max_gap:
        return None
    fraction = (row-a[1])/(b[1]-a[1])
    return float(a[0] + fraction * (b[0]-a[0]))


def main():
    labels_path = OUT / "labels.json"
    signed_path = OUT / "result.json"
    curves = {name: read(path)["tracks"]["ks_silverstone1967"]["diagnostic"]
              for name, path in (("absolute", ABSOLUTE), ("signed", signed_path))}
    labels = read(labels_path)["frames"]
    pixels = read(PIXELS)["frames"]
    camera = read(MOUNT)["views"]["front"]
    rows = []
    for frame, info in pixels.items():
        mark = labels[frame]["target"]
        for point in info["rows"]:
            d = point["station_m"]
            if point["side"] != "right" or d < 16. or point["reference_px"] is None:
                continue
            xr, yr = point["reference_px"]
            depth = d - camera["origin_body_m"][0]
            yf = (.5 + .5 * camera["projection_y"] * camera["origin_body_m"][2] / depth) * 179.
            xm_fixed, xm_map = sample(mark, yf, np.inf), sample(mark, yr, np.inf)
            traced = {name: sample([(x, y) for y, x in values[frame]["trace"][0]], yf)
                      for name, values in curves.items()}
            if xm_fixed is None or xm_map is None or any(v is None for v in traced.values()):
                continue
            recognition = {k: v - xm_fixed for k, v in traced.items()}
            row_shift = xm_fixed - xm_map
            label_to_reference = xm_map - xr
            total = {k: v - xr for k, v in traced.items()}
            for name in traced:
                np.testing.assert_allclose(recognition[name] + row_shift + label_to_reference, total[name])
            rows.append(dict(frame=int(frame), station_m=d, flat_y_px=yf, map_y_px=yr, map_x_px=xr,
                             recognition_dx_px=recognition, row_shift_dx_px=row_shift,
                             label_reference_dx_px=label_to_reference, total_dx_px=total))
    if not rows:
        raise ValueError("No jointly visible annotated stations")
    values = {name: [r[key][mode] for r in rows]
              for key in ("recognition_dx_px", "total_dx_px")
              for mode in curves for name in (f"{key}_{mode}",)}
    values.update({k: [r[k] for r in rows] for k in ("row_shift_dx_px", "label_reference_dx_px")})
    summary = {k: dict(signed_median=float(np.median(v)), absolute_median=float(np.median(np.abs(v))))
               for k, v in values.items()}
    data = dict(status="diagnostic_only", frames=len({r["frame"] for r in rows}), stations=len(rows),
                summary=summary, rows=rows,
                source_sha256={str(p): digest(p) for p in (Path(__file__), MOUNT, PIXELS, ABSOLUTE,
                                                         labels_path, signed_path)})
    (OUT / "pixel-error-decomposition.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    print(data["frames"], data["stations"], summary)


if __name__ == "__main__":
    main()
