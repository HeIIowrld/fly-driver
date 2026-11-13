"""Replay the fixed forward-distance teacher readout and score road clearance."""
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from score_frozen import (
    ARCHIVES, CLI, EVAL, TRANSFER, TRANSFER_CAPTURE, TEACHER, load_sensor, metadata,
    old, reference_poses, digests, sensor_features, lateral_prediction, teacher_points,
)
from train_optimized import BASE, OUT, FROZEN, HORIZONS
from evaluate_teacher import TRACKS, TRACK_DIR, borders_from_spline, boundary_distance

STEP_M = .25
MARGIN_M = 1.1


def classify_clearance(origin, forward, prediction, borders, result):
    transform = np.column_stack((forward,[-forward[1],forward[0]]))
    if boundary_distance(borders,origin) < 0.:
        result["origin_outside"] += 1
        return
    x = np.arange(0., HORIZONS[-1]+STEP_M/2., STEP_M)
    y = np.interp(x, np.r_[0.,HORIZONS], np.r_[0.,prediction])
    points = np.column_stack((x,y)) @ transform.T + origin
    margin = min(boundary_distance(borders,p) for p in points)
    result["origin_inside"] += 1
    result["outside"] += int(margin < 0.)
    result["below_1_1_m"] += int(margin < MARGIN_M)


def evaluate(track, replay, archives, frozen, models, references):
    receipt = replay / "result.json"
    proof = old.read(receipt)
    raw = replay / "sensors.npz"
    bundle = load_sensor(raw,proof["inputs_sha256"])
    rows = metadata(bundle)
    features = sensor_features(bundle)
    poses = reference_poses(archives,track,proof["archive_bundles"],rows)
    selected = frozen["selected_vision"]
    predictions = {name:lateral_prediction(features,name,model) for name,model in models.items()}
    predictions["straight"] = np.zeros((len(rows),len(HORIZONS)))
    teacher = OUT / f"{track}-teacher.npz"
    if old.digest(teacher) != references[track]["teacher_sha256"]:
        raise ValueError("Teacher checksum differs")
    with np.load(teacher,allow_pickle=False) as data:
        reference = data["optimized_xy_m"]
        center = data["center_xy_m"]
    spline = TRACK_DIR / track / TRACKS[track] / "ai/fast_lane.ai"
    borders = borders_from_spline(spline)
    clearance = Counter()
    target = np.full_like(predictions[selected],np.nan)
    center_target = target.copy()
    exported = []
    for i,(row,pose) in enumerate(zip(rows,poses,strict=True)):
        output = predictions[selected][i]
        emitted = np.isfinite(output).all()
        exported.append(dict(
            run=row["run"], frame=row["frame"], generation=row["generation"],
            observation_time_s=row["received_s"], status="available" if emitted else "abstain",
            points=[dict(forward_m=float(x),left_m=float(y) if emitted else None)
                    for x,y in zip(HORIZONS,output,strict=True)]))
        if pose is None:
            continue
        _,_,origin,forward = pose
        if emitted:
            classify_clearance(origin,forward,output,borders,clearance)
        labels = teacher_points(reference,origin,forward,HORIZONS)
        centers = teacher_points(center,origin,forward,HORIZONS)
        if labels is not None:
            target[i] = labels
        if centers is not None:
            center_target[i] = centers
    common = np.isfinite(target) & np.isfinite(center_target)
    for prediction in predictions.values():
        common &= np.isfinite(prediction)
    available = {name:int(np.isfinite(value).all(axis=1).sum()) for name,value in predictions.items()}
    report = {}
    for name, prediction in {**predictions,"offline_center":center_target}.items():
        error = np.where(common,abs(prediction-target),np.nan)
        report[name] = dict(mae_m=np.nanmean(error,axis=0).tolist(),p95_m=np.nanpercentile(error,95,axis=0).tolist())
    file = OUT / f"{track}-optimized-readout.jsonl"
    old.jsonl(file,exported)
    print(track,"common",common.sum(axis=0).tolist(),
          "20m", {name:round(m["mae_m"][-1],3) for name,m in report.items()},
          "clearance",dict(clearance),flush=True)
    return dict(frames=len(rows),common=common.sum(axis=0).tolist(),metrics=report, available=available,
                selected_clearance=dict(clearance), output_sha256=old.digest(file),
                sources=digests((raw,receipt,archives,teacher,spline)))


def main():
    frozen = old.read(FROZEN)
    model_path = OUT / "optimized-readouts.joblib"
    if old.digest(model_path) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen models changed")
    for path, expected in frozen["sources"].items():
        if old.digest(Path(path)) != expected:
            raise ValueError(f"Frozen dependency changed: {path}")
    models = joblib.load(model_path)
    if set(models) != set(frozen["models"]):
        raise ValueError("Frozen model inventory differs")
    inputs = [(track,CLI/("vallelunga-retry" if name=="vallelunga" else name),ARCHIVES[name]) for track,name in EVAL]
    inputs.append(("ks_highlands",TRANSFER/"archive-replay",TRANSFER_CAPTURE/"archive-map.json"))
    references = old.read(TEACHER)["tracks"]
    rows = {track:evaluate(track,replay,archives,frozen,models,references) for track,replay,archives in inputs}
    old.write(OUT/"optimized-readout-evaluation.json",dict(
        selected=frozen["selected_vision"], tracks=rows, horizons_m=HORIZONS.tolist(),
        comparison_kind="existing recordings, including previously evaluated Highlands",
        sources=digests((FROZEN,TEACHER,model_path,BASE/"frozen.json",Path(__file__))),
        source_revision=old.revision(),seed=None,termination="archive_end"))


if __name__ == "__main__":
    main()
