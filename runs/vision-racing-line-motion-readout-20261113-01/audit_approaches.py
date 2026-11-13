"""Audit path anticipation on corner approaches and road clearance on every saved comparison."""
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
from evaluate_readout import ARCHIVES, CLI, EVAL, HORIZONS, MODES, STRAIGHT, digests, error_metrics, load_fixed, metadata, old, predict_all, reference_poses
from fixed_left import keys, load_sensor, reference
from teacher_labels import future_targets
from line_readout import bounded_path, lateral_prediction, sensor_features
from evaluate_live_corner_geometry import load_edges

TRACK_DIR = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks")
LAYOUT = {"ks_vallelunga": "extended_circuit"}
EVALUATION = OUT / "evaluation.json"


def borders_from_spline(path):
    edges, _ = load_edges(path)
    polygons = [np.ascontiguousarray(edge[:,[2,0]], dtype=np.float32) for edge in edges]
    return sorted(polygons, key=cv2.contourArea, reverse=True)


def boundary_distance(borders, position):
    outer, inner = borders
    return min(cv2.pointPolygonTest(outer,tuple(position),True),
               -cv2.pointPolygonTest(inner,tuple(position),True))


def evaluate_clearance(features, model, mode, poses, borders):
    prediction = lateral_prediction(features, mode, model)
    sample_idx = (HORIZONS/.25).astype(int)
    checks = Counter()
    by_point = np.zeros((3,3),dtype=int)
    for i, pose in enumerate(poses):
        if pose is None or not np.isfinite(prediction[i]).all():
            continue
        _,_,position,forward = pose
        rotation = np.column_stack((forward,[-forward[1],forward[0]]))
        path = bounded_path(prediction[i]) @ rotation.T + position
        clearance = np.array([boundary_distance(borders,p) for p in path])
        checks["checked"] += 1
        if clearance[0] < 0.:
            checks["origin_outside"] += 1
            continue
        checks["origin_inside"] += 1
        checks["sampled_curve_outside"] += int((clearance<0.).any())
        checks["sampled_curve_below_1_1m"] += int((clearance<1.1).any())
        by_point += np.column_stack((np.ones(3,dtype=int),clearance[sample_idx]<0.,clearance[sample_idx]<1.1))
    return dict(curve=dict(checks), by_horizon=dict(zip(("5","10","20"),[dict(zip(
        ("checked","outside","below_1_1m"),row.tolist(),strict=True)) for row in by_point],strict=True)))


def main():
    policy, models = load_fixed()
    mode = policy["selected_vision_readout"]
    results = {}
    for track, name in EVAL:
        archive_replay = CLI / ("vallelunga-retry" if name=="vallelunga" else name)
        receipt = archive_replay / "result.json"
        proof = old.read(receipt)
        source = archive_replay / "sensors.npz"
        sensor = load_sensor(source,proof["inputs_sha256"])
        rows = metadata(sensor)
        predictions = predict_all(sensor,models)
        predictions["straight"] = np.broadcast_to(STRAIGHT,(len(rows),*STRAIGHT.shape))
        poses = reference_poses(ARCHIVES[name],track,proof["archive_bundles"],rows)
        target, _ = future_targets(rows,poses)
        labeled, label_file = reference(track)
        actual = {(r["run"],r["frame"]): int(r["label"]) for r in labeled if r["truth"]=="approach"}
        direction = np.array([actual.get(k,0) for k in keys(rows)])
        raw_yaw = sensor["motion"][:,2]
        quiet_yaw = abs(raw_yaw) <= .10
        metrics = error_metrics(target,predictions,dict(
            all_approaches=direction!=0,left=direction==1,right=direction==2,
            left_quiet_yaw=(direction==1)&quiet_yaw,right_quiet_yaw=(direction==2)&quiet_yaw))
        spline = TRACK_DIR / track / LAYOUT.get(track,"") / "ai/fast_lane.ai"
        clearance = evaluate_clearance(sensor_features(sensor),models[mode],mode,poses,borders_from_spline(spline))
        results[track] = dict(approaches=metrics, existing_corner_approaches=dict(Counter(str(d) for d in actual.values())),
                              geometry_clearance=clearance,
                              source_sha256=digests((source,receipt,ARCHIVES[name],label_file,spline)))
        print(track, "left/right approaches", metrics["left"]["20"]["common"],metrics["right"]["20"]["common"],
              "clearance",clearance["curve"],flush=True)
    old.write(OUT / "approaches-and-clearance.json",dict(tracks=results,selected=mode,
             quiet_yaw_abs_max_rad_s=.10, clearance_sampling_step_m=.25, seed=None, source_revision=old.revision(),
             source_sha256=digests((Path(__file__),OUT/"frozen.json",EVALUATION)),
             termination="archive_end"))


if __name__=="__main__":
    main()
