"""Evaluate body-frame racing-line output on saved laps and separate teacher trajectories."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0, str(RUNS / "vision-corner-fixed-left-20261113-01"))
sys.path.insert(0, str(RUNS / "vision-corner-onset-through-line-20261113-01"))
from fixed_left import CLI, EVAL, digests, frames, keys, load_sensor, old
from line_readout import MODES, HORIZONS, bounded_path, constant_body_path, lateral_prediction, metadata, predict, sensor_features
from teacher_labels import future_targets, read_render_pose
from train_vision_teacher_line import edge_polygons

FROZEN = OUT / "frozen.json"
PREVIOUS = RUNS / "vision-corner-onset-through-line-20261113-01"
TEACHER = RUNS / "vision-teacher-line-20261110-02"
ARCHIVES = {
    "silverstone": RUNS / "vision-silverstone1967-independent-capture-20261112-05/archive-map.json",
    "vallelunga": RUNS / "vision-vallelunga-local-corner-capture-20261112-01/archive-map.json",
    "monza": RUNS / "vision-corner-archive-observer-20261112-01/monza-transfer-inputs/archive-map.json",
}
STRAIGHT = np.column_stack((HORIZONS, np.zeros(len(HORIZONS))))


def load_fixed():
    policy = old.read(FROZEN)
    model_file = OUT / "readouts.joblib"
    if old.digest(model_file) != policy["model_sha256"]:
        raise ValueError("Racing line checkpoint changed")
    for path, sha in policy["sources"].items():
        if old.digest(Path(path)) != sha:
            raise ValueError(f"Frozen readout dependency changed: {path}")
    return policy, joblib.load(model_file)


def reference_poses(mapping, track, bundles, expected):
    poses = {}
    for run, folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            if archive.bundle_sha256 != bundles[run]:
                raise ValueError("Camera archive differs from the sensor receipt")
            for raw in archive.read("stream.jsonl").splitlines():
                entry = json.loads(raw)
                key = run, entry["capture_id"]
                if key in poses:
                    raise ValueError("Capture timeline repeats a frame")
                poses[key] = read_render_pose(archive, entry, track)
    if set(poses) != set(keys(expected)):
        raise ValueError("Teacher pose and sensor timelines disagree")
    return [poses[k] for k in keys(expected)]


def predict_all(sensor, models):
    features = sensor_features(sensor)
    return {**{name: predict(features,name,models[name]) for name in MODES},
            "constant_yaw": constant_body_path(features)}


def error_metrics(target, curves, groups):
    full = np.isfinite(target).all(axis=2)
    available = {name: np.isfinite(p).all(axis=2) for name,p in curves.items()}
    comparison = np.logical_and.reduce(list(available.values()))
    output = {}
    for name, group in groups.items():
        result = {}
        for j,h in enumerate(HORIZONS):
            label = full[:,j] & group
            common = label & comparison[:,j]
            measures = {}
            for k,path in curves.items():
                valid = label & available[k][:,j]
                delta = abs(path[:,j]-target[:,j])
                measures[k] = dict(valid=int(valid.sum()), missing=int((label & ~available[k][:,j]).sum()),
                    lateral_mae_m=float(delta[valid,1].mean()) if valid.any() else None,
                    lateral_p95_m=float(np.percentile(delta[valid,1],95)) if valid.any() else None,
                    forward_mae_m=float(delta[valid,0].mean()) if valid.any() else None,
                    common_lateral_mae_m=float(delta[common,1].mean()) if common.any() else None)
            result[str(int(h))] = dict(labeled=int(label.sum()), common=int(common.sum()), methods=measures)
        output[name] = result
    return output


def rolling_metrics(rows, target, previous, estimates):
    result = {}
    for j,h in enumerate(HORIZONS):
        rolling = np.full(len(rows),np.nan)
        for i, (row, point) in enumerate(zip(rows,target[:,j],strict=True)):
            if not np.isfinite(point).all():
                continue
            old_path = previous[row["run"],row["frame"]]["lines"]["geometry"]
            if old_path is not None and old_path["x_m"][0] <= point[0] <= old_path["x_m"][-1]:
                rolling[i] = np.interp(point[0],old_path["x_m"],old_path["y_m"])
        measures = {}
        for name,pred in estimates.items():
            y_at_x = np.full(len(rows),np.nan)
            usable = np.isfinite(pred).all(axis=(1,2)) & np.isfinite(rolling)
            for i in np.flatnonzero(usable):
                path = bounded_path(pred[i,:,1])
                x = target[i,j,0]
                if 0. <= x <= path[-1,0]:
                    y_at_x[i] = np.interp(x,path[:,0],path[:,1])
            same = np.isfinite(y_at_x)
            measures[name] = dict(n=int(same.sum()), old_mae_m=float(abs(rolling[same]-target[same,j,1]).mean()) if same.any() else None,
                                  new_mae_m=float(abs(y_at_x[same]-target[same,j,1]).mean()) if same.any() else None)
        result[str(int(h))] = measures
    return result


def official_monza(rows, paths, teacher_path, camera_path, pose, sensor, models, selected):
    reference, legacy = load_teacher(teacher_path,camera_path)
    index = {k:i for i,k in enumerate(keys(rows))}
    ids = np.array([index[r["episode_id"],r["capture_id"]] for r in reference])
    target = np.array([[(p["x_m"],p["y_m"]) for p in r["points"]] for r in reference])
    match = dict((name,array[ids]) for name,array in paths.items())
    match["camera_speed"] = np.array([[(p["x_m"],p["y_m"]) for p in r["points"]] for r in legacy])
    groups = np.array([r["current_turn"] for r in reference])
    groups = dict(all=np.ones(len(ids),bool),left=groups>0,right=groups<0,straight=groups==0)
    report = error_metrics(target,match,groups)
    borders = edge_polygons(Path("assets/external/ac-teacher-geometry-20261027/monza.csv"))
    # A separate geometry check uses the render anchor for our readout and the
    # same render frame for the baseline; records lacking a pose stay explicit.
    map_report = {}
    selected_lateral = lateral_prediction(sensor_features(sensor),selected,models[selected])
    for name in ("straight","constant_yaw","boundary","boundary_motion","motion", "camera_speed"):
        n = np.zeros(len(HORIZONS),int)
        outside = n.copy()
        teacher_outside = n.copy()
        continuous = 0
        continuous_outside = 0
        for k,i in enumerate(ids):
            if pose[i] is None:
                continue
            _, _, origin, forward = pose[i]
            rotation = np.column_stack((forward,[-forward[1],forward[0]]))
            expected = target[k] @ rotation.T + origin
            def is_out(p):
                outer,inner = borders
                return cv2.pointPolygonTest(outer,tuple(p),False)<0 or cv2.pointPolygonTest(inner,tuple(p),False)>0
            row = match[name][k]
            use = np.isfinite(row).all(axis=1)
            if use.any():
                n += use
                outside[use] += [is_out(p) for p in row[use] @ rotation.T + origin]
                teacher_outside[use] += [is_out(p) for p in expected[use]]
            if name == selected and np.isfinite(selected_lateral[i]).all():
                path = bounded_path(selected_lateral[i])
                continuous += 1
                continuous_outside += int(any(is_out(p) for p in path @ rotation.T + origin))
        map_report[name] = dict(checked=n.tolist(), point_outside=outside.tolist(), teacher_outside=teacher_outside.tolist(),
                                sampled_curve_checked=continuous, sampled_curve_outside=continuous_outside)
    return dict(frames=len(ids), metric="fixed arc length point error", metrics=report,
                render_anchor_boundary=map_report)


def load_teacher(teacher_path,camera_path):
    checksums = old.read(TEACHER / "artifact-sha256.json")
    ref = [r for r in frames(teacher_path,checksums[teacher_path.name])
           if r["split"]=="test" and r["episode_id"]=="fresh-monza-develop-01"]
    legacy = frames(camera_path,checksums[camera_path.name])
    if [(r["episode_id"],r["observation_time_s"]) for r in ref] != [(r["episode_id"],r["observation_time_s"]) for r in legacy]:
        raise ValueError("Legacy Monza prediction keys differ from the qualified lap")
    return ref, legacy


def export_rows(path, rows, sensor, lateral):
    with path.open("x",encoding="utf-8") as stream:
        for i,row in enumerate(rows):
            curve = bounded_path(lateral[i])
            use = curve is not None
            positions = curve[(HORIZONS/.25).astype(int)] if use else [[None,None]] * len(HORIZONS)
            points = [dict(s_m=float(h),x_m=float(x) if use else None,y_m=float(y) if use else None,valid=use)
                      for h,(x,y) in zip(HORIZONS,positions,strict=True)]
            speed = sensor["motion"][i,1]
            record = dict(schema="flybrain.racing_line.v1", episode_id=row["run"], frame=row["frame"],
                          generation=row["generation"], clock_id="host-qpc-camera-received",
                          observation_time_s=row["received_s"], ego_frame="body_xy_forward_left",
                          origin="vehicle_reference_point", source_speed_mps=float(speed) if np.isfinite(speed) else None,
                          status="available" if use else "abstain", points=points)
            stream.write(json.dumps(record,allow_nan=False)+"\n")


def main():
    policy, models = load_fixed()
    selected = policy["selected_vision_readout"]
    line_receipt = PREVIOUS / "line-result.json"
    previous = old.read(line_receipt)
    outputs = {}
    for track,name in EVAL:
        directory = CLI / ("vallelunga-retry" if name=="vallelunga" else name)
        receipt = directory / "result.json"
        archived = old.read(receipt)
        path = directory / "sensors.npz"
        sensor = load_sensor(path,archived["inputs_sha256"])
        rows = metadata(sensor)
        forecasts = predict_all(sensor,models)
        forecasts["straight"] = np.broadcast_to(STRAIGHT,(len(rows),*STRAIGHT.shape))
        export = OUT / f"{track}-line.jsonl"
        export_rows(export,rows,sensor,lateral_prediction(sensor_features(sensor),selected,models[selected]))
        poses = reference_poses(ARCHIVES[name],track,archived["archive_bundles"],rows)
        target, _ = future_targets(rows,poses)
        long_y = target[:,-1,1]
        sets = dict(all=np.ones(len(rows),bool), left=long_y>=1.,right=long_y<=-1.,straight=abs(long_y)<1.)
        results = error_metrics(target,forecasts,sets)
        previous_path = PREVIOUS / f"{track}-lines.jsonl"
        previous_rows = frames(previous_path,previous["tracks"][track]["lines_sha256"])
        if keys(rows) != keys(previous_rows):
            raise ValueError("Historical line frame identity differs from the sensors")
        matched = rolling_metrics(rows,target,dict(zip(keys(rows),previous_rows,strict=True)),
                                  {name:forecasts[name] for name in MODES})
        data = dict(frames=len(rows), label_kind="verified recorded future travel", output_sha256=old.digest(export),
                    line_abstentions=int(np.isnan(forecasts[selected][:,0,0]).sum()), point_metrics=results,
                    previous_rolling_at_same_forward_x=matched)
        sources = {path,receipt,ARCHIVES[name],previous_path}
        if name=="monza":
            teacher_path = TEACHER / "teacher-labels.jsonl"
            camera_path = TEACHER / "monza_test-vision_speed.jsonl"
            data["official_evaluation_lap"] = official_monza(rows,forecasts,teacher_path,camera_path,poses,sensor,models,selected)
            sources.update((teacher_path,camera_path,Path("assets/external/ac-teacher-geometry-20261027/monza.csv")))
        data["source_sha256"] = digests(sources)
        outputs[track] = data
        print(track,"selected",selected,"out of",len(rows), "abstentions",data["line_abstentions"],
              {g:{k:round(v["common_lateral_mae_m"],3) if v["common_lateral_mae_m"] is not None else None
                  for k,v in results[g]["20"]["methods"].items()} for g in sets},flush=True)
    old.write(OUT / "evaluation.json",dict(tracks=outputs, selected=selected, source_revision=old.revision(),
              seed=None, source_sha256=digests((Path(__file__),FROZEN,line_receipt)),
              termination="archive_end"))


if __name__ == "__main__":
    main()
