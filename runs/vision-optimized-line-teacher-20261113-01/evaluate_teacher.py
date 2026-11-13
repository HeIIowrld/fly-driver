"""Evaluate a fixed low-curvature teacher and offline nominal speed envelope."""
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / "vision-racing-line-motion-readout-20261113-01"
sys.path.insert(0, str(BASE))
from audit_approaches import TRACK_DIR, borders_from_spline, boundary_distance
from evaluate_readout import old, digests
from evaluate_live_corner_geometry import load_edges
from optimize_line import MARGIN_M, optimize_line

TRACKS = {
    "imola": "",
    "ks_laguna_seca": "",
    "ks_zandvoort": "",
    "mugello": "",
    "ks_silverstone1967": "",
    "ks_vallelunga": "extended_circuit",
    "monza": "",
    "ks_highlands": "layout_short",
}
LATERAL_ACCEL_MPS2 = 11.0
BRAKE_ACCEL_MPS2 = 9.0
DRIVE_ACCEL_MPS2 = 4.0
MAX_SPEED_MPS = 72.0


def nominal_lap(path):
    segments = np.roll(path, -1, axis=0) - path
    lengths = np.linalg.norm(segments, axis=1)
    heading = np.arctan2(segments[:, 1], segments[:, 0])
    angles = np.roll(heading, -1) - heading
    curvature = abs(np.arctan2(np.sin(angles), np.cos(angles))) / (
        (lengths + np.roll(lengths, -1)) * 0.5)
    corner_speed_squared = np.minimum(MAX_SPEED_MPS**2, LATERAL_ACCEL_MPS2 / np.maximum(curvature, 1e-8))
    speed_squared = corner_speed_squared.copy()
    for _ in range(2):
        for i in range(len(path)):
            j = (i + 1) % len(path)
            speed_squared[j] = min(speed_squared[j], speed_squared[i] + 2 * DRIVE_ACCEL_MPS2 * lengths[i])
        for i in range(len(path)-1, -1, -1):
            j = (i + 1) % len(path)
            speed_squared[i] = min(speed_squared[i], speed_squared[j] + 2 * BRAKE_ACCEL_MPS2 * lengths[i])
    speed = np.sqrt(speed_squared)
    return dict(time_s=float(np.sum(2 * lengths / (speed + np.roll(speed, -1)))),
                distance_m=float(lengths.sum()), curvature_p95_per_m=float(np.quantile(curvature, 0.95)),
                curvature_max_per_m=float(curvature.max()), min_speed_mps=float(speed.min()),
                speed_max_mps=float(speed.max()))


def clearance(path, borders):
    # Include subsegment points to test between the optimization stations.
    pieces = np.linalg.norm(np.roll(path, -1, axis=0)-path, axis=1)
    arc = np.r_[0., np.cumsum(pieces)]
    sample = np.arange(0., arc[-1], .25)
    closed = np.r_[path,path[:1]]
    samples = np.column_stack([np.interp(sample, arc, column) for column in closed.T])
    values = np.array([boundary_distance(borders, point) for point in samples])
    return dict(samples=len(values), outside=int((values < 0.).sum()),
                below_margin=int((values < MARGIN_M).sum()), min_clearance_m=float(values.min()))


def main():
    inputs = {Path(__file__), Path(optimize_line.__code__.co_filename)}
    report = {}
    for name, layout in TRACKS.items():
        spline = TRACK_DIR / name / layout / "ai/fast_lane.ai"
        inputs.add(spline)
        edges, _ = load_edges(spline)
        path, center, fit = optimize_line(edges)
        borders = borders_from_spline(spline)
        path_file = OUT / f"{name}-teacher.npz"
        np.savez_compressed(path_file, optimized_xy_m=path, center_xy_m=center)
        report[name] = dict(layout=layout, fit=fit, teacher_sha256=old.digest(path_file),
                            center=nominal_lap(center), optimized=nominal_lap(path),
                            center_clearance=clearance(center,borders), optimized_clearance=clearance(path,borders))
        result = report[name]
        print(name, "nominal seconds", result["center"]["time_s"], result["optimized"]["time_s"],
              "clearance", result["optimized_clearance"], flush=True)
    old.write(OUT / "teacher-evaluation.json", dict(
        tracks=report, nominal_speed_model=dict(lateral_accel_mps2=LATERAL_ACCEL_MPS2,
        brake_accel_mps2=BRAKE_ACCEL_MPS2, drive_accel_mps2=DRIVE_ACCEL_MPS2,
        max_speed_mps=MAX_SPEED_MPS), sources=digests(inputs), seed=None,
        source_revision=old.revision(), termination="offline_geometry_complete"))


if __name__ == "__main__":
    main()
