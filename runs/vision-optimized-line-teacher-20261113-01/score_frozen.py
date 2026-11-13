"""Score frozen path outputs against a low-curvature offline reference."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
BASE = RUNS / "vision-racing-line-motion-readout-20261113-01"
sys.path.insert(0, str(BASE))
from evaluate_readout import (
    ARCHIVES, CLI, EVAL, digests, load_fixed, load_sensor, metadata, old, reference_poses,
)
from line_readout import CAMERA_STEP, HORIZONS, bounded_path, lateral_prediction, sensor_features
from teacher_labels import at

TRANSFER = RUNS / "vision-highlands-short-fixed-corner-evaluation-20261113-01"
TRANSFER_CAPTURE = RUNS / "vision-highlands-short-fixed-corner-capture-20261113-06"
TEACHER = OUT / "teacher-evaluation.json"


def teacher_points(line, origin, forward, x_stations):
    end = np.roll(line, -1, axis=0)
    delta = end-line
    lengths = np.linalg.norm(delta, axis=1)
    offset = origin-line
    proportion = np.clip(np.sum(offset*delta, axis=1)/lengths**2,0.,1.)
    closest = line + proportion[:,None]*delta
    separation = np.linalg.norm(origin-closest,axis=1)
    candidates = (delta @ forward > .65 * lengths) & (separation < 15.)
    if not candidates.any():
        return None
    start = int(np.argmin(np.where(candidates,separation,np.inf)))
    indices = (start + np.arange(min(len(line),30))) % len(line)
    arc = np.r_[0.,np.cumsum(lengths[indices])]
    path = np.r_[line[indices],end[indices[-1:]]]
    query = np.arange(proportion[start]*lengths[start], arc[-1], CAMERA_STEP)
    if len(query) < 2:
        return None
    transform = np.column_stack((forward,[-forward[1],forward[0]]))
    local = (at(arc,path,query)-origin) @ transform
    nonforward = np.flatnonzero(np.diff(local[:,0]) <= 0.)
    if len(nonforward):
        local = local[:nonforward[0]+1]
    y = np.full(len(x_stations), np.nan)
    visible = (x_stations >= local[0,0]) & (x_stations <= local[-1,0])
    y[visible] = np.interp(x_stations[visible],local[:,0],local[:,1])
    return y


def evaluate(track, replay, archives, teacher, models, mode):
    source = replay / "sensors.npz"
    receipt = replay / "result.json"
    proof = old.read(receipt)
    sensors = load_sensor(source,proof["inputs_sha256"])
    rows = metadata(sensors)
    poses = reference_poses(archives,track,proof["archive_bundles"],rows)
    path = OUT / f"{track}-teacher.npz"
    if old.digest(path) != teacher[track]["teacher_sha256"]:
        raise ValueError("Reference checksum differs")
    with np.load(path, allow_pickle=False) as data:
        reference = data["optimized_xy_m"]
    features = sensor_features(sensors)
    predictions = {name:lateral_prediction(features,name,models[name])
                   for name in (mode,"body","boundary_motion")}
    errors = {name:[] for name in (*predictions,"straight")}
    evidence = Counter()
    ids = (HORIZONS/CAMERA_STEP).astype(int)
    for i, pose in enumerate(poses):
        if pose is None or not all(np.isfinite(prediction[i]).all() for prediction in predictions.values()):
            continue
        evidence["common_path_and_pose"] += 1
        _,_,origin,forward = pose
        # Match the selected path's longitudinal coordinates for every method.
        selected = bounded_path(predictions[mode][i])
        requested = selected[ids,0]
        paths = {name:bounded_path(prediction[i]) for name,prediction in predictions.items()}
        lateral = {name:np.interp(requested,points[:,0],points[:,1],left=np.nan,right=np.nan)
                   for name,points in paths.items()}
        lateral["straight"] = np.zeros(len(ids))
        ref = teacher_points(reference,origin,forward,requested)
        if ref is None:
            continue
        valid = np.isfinite(ref) & np.logical_and.reduce([np.isfinite(y) for y in lateral.values()])
        for name,y in lateral.items():
            errors[name].append(np.where(valid,abs(y-ref),np.nan))
    report = {}
    for name, values in errors.items():
        array = np.asarray(values)
        report[name] = dict(n=np.isfinite(array).sum(axis=0).tolist(),
                            lateral_mae_m=np.nanmean(array,axis=0).tolist(),
                            lateral_p95_m=np.nanpercentile(array,95,axis=0).tolist())
    print(track, report, flush=True)
    return dict(total_frames=len(rows), matches=dict(evidence), methods=report,
                source_sha256=digests((source,receipt,archives,path)))


def main():
    policy, models = load_fixed()
    mode = policy["selected_vision_readout"]
    teacher = old.read(TEACHER)["tracks"]
    inputs = [(t,CLI/("vallelunga-retry" if name=="vallelunga" else name),ARCHIVES[name]) for t,name in EVAL]
    inputs.append(("ks_highlands",TRANSFER/"archive-replay",TRANSFER_CAPTURE/"archive-map.json"))
    records = {track:evaluate(track,replay,archives,teacher,models,mode) for track,replay,archives in inputs}
    old.write(OUT/"readout-comparison.json",dict(tracks=records, selected=mode,
              metric="lateral distance to the low-curvature line at equal forward stations",
              forward_station_source="selected frozen path at 5/10/20 m arc",
              sources=digests((Path(__file__),TEACHER,BASE/"frozen.json")), seed=None,
              source_revision=old.revision(), termination="archive_end"))


if __name__ == "__main__":
    main()
