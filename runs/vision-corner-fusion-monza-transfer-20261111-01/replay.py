"""Replay the development-selected fusion on Monza's held physical groups."""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from infer_frozen_next_corner import load_readout
from train_next_corner import BASE, MONZA
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
runs = out.parent
temp = runs/"vision-corner-temporal-boundaries-20261111-01"
uncertainty = runs/"vision-corner-boundary-uncertainty-20261111-01"
fused = runs/"vision-corner-fused-confirmation-20261111-01"
factor = runs/"vision-corner-factorized-20261111-01"
monza = runs/"vision-monza-two-lap-presence-20261111-01"
source = runs/"vision-magione-smoothed-sustained-onset-20261111-01"/"audit.py"
config = Path("configs/ac-csp-exterior-front-v3-retina.json")
blend = runs/"vision-corner-blended-candidate-20261111-02"/"pins.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load_module("temporal_boundary_replay", temp/"replay.py")
shape = load_module("boundary_uncertainty", uncertainty/"audit.py")
selection = read(fused/"result.json")["lowest_false_with_joint_and_left_gain"]["setting"]
assert selection["geometry"] == "whole_shape_min_0.002" and selection["model"] == "contrast-10000"
assert selection["distance_source"] == "ordinal" and not selection["require_side"]
assert selection["distance_limit"] is None
tree = ast.parse(source.read_text(encoding="utf-8-sig"))
detector = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SurfaceContrast")
exec(compile(ast.Module(body=[detector], type_ignores=[]), str(source), "exec"))
near = SurfaceContrast(read(config), near=True)
far = SurfaceContrast(read(config))
cv2.setNumThreads(1)
inputs = [Path(__file__), temp/"replay.py", uncertainty/"audit.py", fused/"result.json", source, config, blend]

receipt = read(monza/"flyvis-result.json")
monza_rows = replay.read_rows(monza/"rows.jsonl", receipt["rows_sha256"])
fit_groups = {r["physical_group"] for r in monza_rows if r["split"] == "train"}
selected = {replay.key(r): r for r in monza_rows if r["split"] == "validation"}
assert len(selected) == 100 and fit_groups.isdisjoint(r["physical_group"] for r in selected.values())
features_file = monza/"features.npz"
assert digest(features_file) == receipt["features_sha256"]
factor_receipt = read(factor/"result.json")
scores = {replay.key(r): r for r in replay.read_rows(factor/"predictions.jsonl", factor_receipt["prediction_sha256"])
          if r["model"] == selection["model"] and r["cohort"] == "monza_validation"}
assert scores.keys() == selected.keys()
inputs.extend((monza/"flyvis-result.json", monza/"rows.jsonl", features_file,
               factor/"result.json", factor/"predictions.jsonl"))

def decode(archive, view):
    image = cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable recorded camera image")
    return image


def capture_replay():
    for run, sub in MONZA:
        receipt_path = BASE/run/sub/"result.json"
        inputs.append(receipt_path)
        expected = read(receipt_path)["camera_bundle_sha256"]
        selected_frames = {frame for name, frame in selected if name == run}
        # The predictor needs three consecutive decisions and up to .24 s of earlier
        # geometry. Decode surrounding captures without reading their labels.
        wanted = {f for frame in selected_frames for f in range(frame-10, frame+1)}
        with CameraArchive(BASE/run/"demonstration"/"camera"/"vision") as archive:
            assert archive.bundle_sha256 == expected
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                frame = item["capture_id"]
                if frame not in wanted:
                    continue
                entry = dict(run=run, frame=frame, generation=item["generation"], motion=None, sections=None)
                views = {v["view_id"]: v for v in item["views"]}
                if set(views) == {"front", "rear", "left", "right"} and not (
                        item.get("ideal_line_shown") or item.get("aid_probe_only")):
                    meta = archive.json(Path(item["file"]).name)
                    assert meta["index"] == frame and meta["track_id"] == "monza"
                    body = max(meta["views"], key=lambda v: v["render_call_finished_clock"]).get("source_body_after_render")
                    if body and body.get("physics_available") and body.get("local_angular_velocity") is not None:
                        values = [body.get("source_clock_finished"), body.get("speed_m_s"),
                                  body["local_angular_velocity"][1]]
                        if all(v is not None and np.isfinite(v) for v in values) and values[1] >= 0:
                            entry["motion"] = values
                    images = {name: decode(archive, v) for name, v in views.items()}
                    sections = near.detect(images)+far.detect({"front": images["front"]})[1:]
                    entry["sections"] = [[s["forward_x_m"], s["right_y_m"], s["left_y_m"]]
                                         for s in sections if s["valid"]]
                yield entry


observed = list(capture_replay())
raw_geometry = list(replay.predictions(observed))
confirmed_source = []
for proof, center, scatter, values in shape.recovered(observed, raw_geometry):
    raw = proof["curvature"].get("history_0.24", [])
    samples = shape.whole_shape_bounds(raw, values)["whole_shape_min"] if center is not None else []
    confirmed_source.append(dict(proof, curvature={"history_0.24": samples}))
confirmed = replay.confirmed_predictions(confirmed_source, .002, 3, 1.)
geometric_only = replay.confirmed_predictions(raw_geometry, .002, 3, 1.)
raw_index = {replay.key(r): r for r in raw_geometry}
shape_index = {replay.key(r): r for r in confirmed_source}
motion_index = {replay.key(r): r["motion"] for r in observed}

pins = read(blend)
predict, gate = load_readout(pins)
for spec in pins["models"].values():
    inputs.append(Path(spec["path"]))

outputs = []
stage_counts = Counter()
with np.load(features_file, allow_pickle=False) as data:
    a, b, mirrored = (data[k] for k in ("direct", "reflected", "reflected_available"))
    for i, row in enumerate(monza_rows):
        key = replay.key(row)
        if key not in selected:
            continue
        side, distance, presence = (float(x[0]) for x in predict(a[i:i+1], b[i:i+1] if mirrored[i] else None))
        actual = row["direction"]
        truth_distance = row["distance"]
        truth = ("straight" if actual == 0 else "approach" if 4 <= truth_distance <= 20 else "out_of_range")
        record = dict(run=row["run"], frame=row["frame"], received_s=row["time"], generation=row["run"],
                      truth=truth, label=str(actual), physical_corner=row["event"],
                      truth_distance_m=truth_distance,
                      truth_endpoints_m=[truth_distance] if truth_distance is not None else [])
        alert = side and presence >= gate and distance <= 20
        baseline = replay.rescore(record, (int(side), distance) if alert else
                                  (0, None) if not side else None)
        choice = confirmed.get(key)
        evidence = scores[key]
        add = not alert and choice and evidence["presence_score"] >= selection["threshold"] and 0 <= evidence["distance_m"] <= 20
        fused_row = replay.rescore(record, (choice[0], evidence["distance_m"])) if add else baseline
        if not alert:
            stage_counts["baseline_without_alert"] += 1
            stage_counts["raw_temporal"] += bool(geometric_only.get(key))
            stage_counts["whole_shape_temporal"] += bool(choice)
            stage_counts["fused_add"] += bool(add)
        if truth == "approach" and not alert:
            stage_counts["missed_approach"] += 1
            stage_counts["missed_approach_with_raw_geometry"] += bool(geometric_only.get(key))
            stage_counts["missed_approach_with_whole_shape"] += bool(choice)
            raw_now = replay.decide(raw_index[key]["curvature"].get("history_0.24", []), .002)
            shaped_now = replay.decide(shape_index[key]["curvature"].get("history_0.24", []), .002)
            stage_counts["missed_with_current_temporal_candidate"] += bool(raw_now and raw_now[0])
            stage_counts["missed_with_current_whole_shape_candidate"] += bool(shaped_now and shaped_now[0])
        outputs.append(dict(run=row["run"], frame=row["frame"], raw=geometric_only.get(key),
                            confirmed=choice, visual_score=evidence["presence_score"],
                            motion=motion_index[key], travel_m=raw_index[key]["travel_m"],
                            raw_before_confirmation=replay.decide(raw_index[key]["curvature"].get("history_0.24", []), .002),
                            shape_before_confirmation=replay.decide(shape_index[key]["curvature"].get("history_0.24", []), .002),
                            visual_distance_m=evidence["distance_m"], parent=baseline, fused=fused_row))

assert len(outputs) == 100 and selected.keys() <= confirmed.keys()
summaries = {name: replay.reports([r[name] for r in outputs]) for name in ("parent", "fused")}
jsonl(out/"rows.jsonl", outputs)
jsonl(out/"observations.jsonl", observed)
jsonl(out/"geometry.jsonl", [dict(raw=r, shape=s) for r, s in zip(raw_geometry, confirmed_source, strict=True)])
write(out/"result.json", dict(setting=selection, counts=dict(stage_counts), frames=len(outputs),
    held_physical_groups=sorted({r["physical_group"] for r in selected.values()}),
    approach_events=sorted({r["event"] for r in selected.values() if r["distance"] and 4 <= r["distance"] <= 20}),
    coverage=dict(decoded_history=len(observed), body_motion=sum(r["motion"] is not None for r in observed),
                  geometry=sum(r["sections"] is not None for r in observed)),
    summary={k:replay.compact(v) for k,v in summaries.items()},
    paired=replay.matched_changes(summaries["parent"], summaries["fused"]),
    seed=None, candidate_promoted=False, termination="held_development_archive_end",
    rows_sha256=digest(out/"rows.jsonl"), observations_sha256=digest(out/"observations.jsonl"),
    geometry_sha256=digest(out/"geometry.jsonl"), inputs={str(p):digest(p) for p in inputs}))
print(read(out/"result.json")["counts"])
print(read(out/"result.json")["summary"])
