"""Measure local line coverage and lateral error against a qualified saved teacher lap."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0, str(RUNS / "vision-corner-onset-through-line-20261113-01"))
from dense_onset import CLI, frames, old

LINE = RUNS / "vision-corner-onset-through-line-20261113-01"
TEACHER = RUNS / "vision-teacher-line-20261110-02"
LAYOUT = "fresh-monza-develop-01"
METHODS = ("classic", "geometry", "preview", "camera_speed", "straight")
HORIZONS = (5., 10., 20.)


def teacher_data():
    artifact = old.read(TEACHER / "artifact-sha256.json")
    labels_path = TEACHER / "teacher-labels.jsonl"
    camera_path = TEACHER / "monza_test-vision_speed.jsonl"
    label_rows = frames(labels_path, artifact[labels_path.name])
    labels = [r for r in label_rows if r["episode_id"] == LAYOUT and r["split"] == "test"]
    camera = frames(camera_path, artifact[camera_path.name])
    keys = [(r["episode_id"], r["observation_time_s"]) for r in labels]
    if keys != [(r["episode_id"], r["observation_time_s"]) for r in camera] or len(keys) != len(set(keys)):
        raise ValueError("Teacher and held-lap video predictions do not align")
    return labels, camera, labels_path, camera_path


def at_x(line, x):
    if line is None or not line["x_m"][0] <= x <= line["x_m"][-1]:
        return None
    return float(np.interp(x, line["x_m"], line["y_m"]))


def evaluate():
    labels, camera, labels_path, camera_path = teacher_data()
    line_path = LINE / "monza-lines.jsonl"
    line_receipt = LINE / "line-result.json"
    lines = frames(line_path, old.read(line_receipt)["tracks"]["monza"]["lines_sha256"])
    by_key = {(row["run"], row["frame"]): row for row in lines}
    if len(lines) != len(by_key):
        raise ValueError("Line replay has duplicate frames")
    evidence = []
    for ref, direct in zip(labels, camera, strict=True):
        row = by_key[ref["episode_id"], ref["capture_id"]]
        turn = ref["current_turn"]
        group = "left" if turn > 0 else "right" if turn < 0 else "straight"
        for point, camera_point in zip(ref["points"], direct["points"], strict=True):
            if not point["valid"]:
                continue
            s, x, y = (point[name] for name in ("s_m", "x_m", "y_m"))
            if camera_point["s_m"] != s:
                raise ValueError("Prediction and label distance grids differ")
            predictions = {name: at_x(row["lines"][name], x) for name in METHODS[:3]}
            # Compare the learned camera model at the same forward cross-section.
            # Its origin is part of the declared output contract.
            direct_points = [p for p in direct["points"] if p["valid"]]
            direct_x = [0., *(p["x_m"] for p in direct_points)]
            direct_y = [0., *(p["y_m"] for p in direct_points)]
            predictions["camera_speed"] = at_x(dict(x_m=direct_x, y_m=direct_y), x) if np.all(np.diff(direct_x)>0) else None
            predictions["straight"] = 0.
            evidence.append(dict(run=row["run"], frame=row["frame"], group=group, teacher_arc_m=s,
                                 teacher_x_m=x, teacher_y_m=y,
                                 errors_m={name: abs(value-y) if value is not None else None
                                           for name, value in predictions.items()}))
    reports = {}
    for group in ("all", "left", "right", "straight"):
        reports[group] = {}
        for horizon in HORIZONS:
            rows = [row["errors_m"] for row in evidence if row["teacher_arc_m"] == horizon
                    and (group == "all" or row["group"] == group)]
            common = [row for row in rows if all(v is not None for v in row.values())]
            rolling_common = [row for row in rows if all(row[k] is not None for k in ("geometry","preview"))]
            result = dict(total=len(rows), all_methods_common=len(common), rolling_common=len(rolling_common), methods={})
            for name in METHODS:
                errors = [row[name] for row in rows if row[name] is not None]
                matched = [row[name] for row in common]
                shared_rolling = [row[name] for row in rolling_common if row[name] is not None]
                result["methods"][name] = dict(
                    valid=len(errors), mae_m=float(np.mean(errors)) if errors else None,
                    p95_m=float(np.percentile(errors, 95)) if errors else None,
                    all_methods_common_mae_m=float(np.mean(matched)) if matched else None,
                    on_rolling_n=len(shared_rolling),
                    on_rolling_mae_m=float(np.mean(shared_rolling)) if shared_rolling else None)
            reports[group][str(horizon)] = result
    trace = OUT / "monza-teacher-line-cross-sections.jsonl"
    old.jsonl(trace, evidence)
    old.write(OUT / "monza-teacher-line.json", dict(
        evaluation_run=LAYOUT, frames=len(labels), results=reports,
        frame_groups=dict(Counter(r["group"] for r in evidence if r["teacher_arc_m"]==5.)),
        coordinate="lateral error at the teacher point's ego forward cross section",
        reference_is_saved_qualified_teacher_trajectory=True,
        model_source_revision=old.revision(), seed=None, rows_sha256=old.digest(trace),
        source_sha256={str(p): old.digest(p) for p in (line_receipt,line_path,labels_path,camera_path,Path(__file__))},
        termination="archive_end"))
    for horizon, result in reports["all"].items():
        print(horizon, result, flush=True)


if __name__ == "__main__":
    evaluate()
