"""Score the existing image traces against configured surface edges."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np

from kn5_surfaces import OUT, digest

RUNS = OUT.parent
CONTROL = RUNS / "vision-corner-image-row-trace-20261112-01" / "supported" / "result.json"
SIGNED = RUNS / "vision-corner-visible-line-labels-20261112-01" / "result.json"
LABELS = OUT / "surface-labels.json"
MOUNT = RUNS.parent / "configs/ac-csp-exterior-front-v3-retina.json"
TRACK = "ks_silverstone1967"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def position(curve, row):
    low = int(np.floor(row))
    high = int(np.ceil(row))
    if low not in curve or high not in curve:
        return None
    return curve[low] + (row-low)*(curve[high]-curve[low])


def main():
    data = read(LABELS)
    paths = {"absolute": CONTROL, "signed": SIGNED}
    traces = {k: read(p)["tracks"][TRACK]["diagnostic"] for k,p in paths.items()}
    assert set(data["labels"]) == set(traces["absolute"]) == set(traces["signed"])
    camera = read(MOUNT)["views"]["front"]
    by_key = defaultdict(list)
    counts = defaultdict(int)
    details = []
    for frame, label in data["labels"].items():
        curves = {k: [dict(side) for side in r[frame]["trace"]] for k,r in traces.items()}
        for p in label["labels"]:
            if p["station_m"] < 16. or not p["witnessed"] or p["pixel"] is None:
                continue
            side = p["side"]
            index = ("right", "left").index(side)
            depth = p["station_m"] - camera["origin_body_m"][0]
            scaling = .5 * 319. * camera["projection_x"] / depth
            flat_y = (.5 + .5 * camera["projection_y"] * camera["origin_body_m"][2] / depth) * 179.
            true_x, true_y = p["pixel"]
            for sample, y in (("flat_row", flat_y), ("reference_row", true_y)):
                by_trace = {k: position(c[index], y) for k,c in curves.items()}
                for k,x in by_trace.items():
                    if x is not None:
                        counts[f"{side}_{sample}_{k}"] += 1
                if any(x is None for x in by_trace.values()):
                    continue
                for k,x in by_trace.items():
                    pred = camera["origin_body_m"][1] + (159.5-x)/scaling
                    delta = pred - p["lateral_m"]
                    by_key[side, sample, k, "lateral_m"].append(abs(delta))
                    if sample == "reference_row":
                        by_key[side, sample, k, "same_image_row_px"].append(abs(x-true_x))
                details.append(dict(frame=int(frame), station_m=p["station_m"], side=side, sample=sample,
                                    trace_x_px=by_trace, physical_px=p["pixel"], physical_m=p["lateral_m"]))
    metrics = {"_".join(k): dict(points=len(a), median=float(np.median(a))) for k,a in by_key.items()}
    sources = (Path(__file__), LABELS, MOUNT, *paths.values())
    report = dict(status="diagnostic_only", seed=None, source_revision=data["source_revision"],
                  point_counts=dict(counts), paired_metrics=metrics,
                  source_sha256={str(p): digest(p) for p in sources}, details=details)
    (OUT / "existing-trace-physical-score.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
