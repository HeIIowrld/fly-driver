"""Replay projected-tangent boundary selection on Silverstone and Spa."""

import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from corner_edge_tangent import complete, front_normals
from corner_fusion_boundaries import curvature, decide, read_motion, stations
from image_normal import FIELDS, choose, edge_scores
from paired_motion import inputs
import profile_replay as profile
from relative_corner_boundaries import SurfaceContrast
from replay_learned import baseline_data, publish, signals
from score_corner_captures import MIN_SPEED_MPS, summarize
from train_vision_teacher_line import digest, jsonl, read, write
from flybrain_drive.ac_camera_archive import CameraArchive

OUT = Path(__file__).parent
RUNS = OUT.parent
LOCK = RUNS / "vision-vallelunga-local-corner-protocol-20261112-01"
ORIGINAL = RUNS / "vision-corner-local-curvature-20261112-01"
MANUAL = RUNS / "vision-silverstone-visible-edges-20261112-01/manual-right-boundary.json"
GEOMETRY = ("paired", "normal", "bounded_normal")
CONFIRM = {"edge": ("boundaries", "ranked_local_boundary_hold"),
           "t4t5": ("boundaries_temporal_t4t5", "ranked_local_t4t5_hold")}


def key(row):
    return row["run"], row["frame"]


def distance_to_mark(points, mark):
    """Perpendicular distance, evaluated only along the annotated x span."""
    points = points[(points[:, 0] >= mark[0, 0]) & (points[:, 0] <= mark[-1, 0])]
    if not len(points):
        return []
    direction = np.diff(mark, axis=0)
    delta = points[:, None, :] - mark[None, :-1, :]
    t = np.clip(np.sum(delta * direction, axis=-1) / np.sum(direction**2, axis=-1), 0., 1.)
    return np.linalg.norm(delta - t[..., None] * direction, axis=-1).min(axis=1).tolist()


def totals(reports):
    by_side = {side: sum(r["by_direction"][side]["direction_hits"] for r in reports.values())
               for side in ("left", "right")}
    stable = {(e["physical_corner"], e["label"]) for r in reports.values() for e in r["events"]
              if e["first_stable_frame"] is not None}
    return dict(**by_side, joint=sum(s["joint_hits"] for r in reports.values() for s in r["by_direction"].values()),
                outside=sum(r["false_alerts"]["outside_band"]["frames"] for r in reports.values()),
                in_corner_alert=sum(sum(r["truth_by_output"].get("in_corner", {}).get(side, 0)
                                        for side in ("left", "right")) for r in reports.values()),
                stable=len(stable), stable_right=sum(side == "2" for _, side in stable),
                frames=sum(r["frames"] for r in reports.values()))


def run(track, mount, weights, hashes):
    group, _, mapping, bundles = baseline_data(track, hashes)
    input_groups = inputs(track, hashes)
    known = read(ORIGINAL / "result.json")["tracks"][track]
    source = ORIGINAL / f"{track}-frames.jsonl"
    sections = ORIGINAL / f"{track}-sections.npz"
    assert digest(source) == known["rows_sha256"] and digest(sections) == known["sections_sha256"]
    hashes.update((source, sections))
    old = [json.loads(l) for l in source.open()]
    truth = {key(r): r for r in group["rows"]}
    assert list(truth) == [key(r) for r in old]
    old = {key(r): r for r in old}
    with np.load(sections, allow_pickle=False) as array:
        assert list(truth) == list(zip(array["run"], array["frame"], strict=True))
        original_edges = dict(zip(truth, array["sections"][:, 0], strict=True))
    profiles = {}
    for name, (mode, _) in CONFIRM.items():
        data = input_groups[mode]
        assert list(truth) == [key(r) for r in data["rows"]]
        profiles[name] = dict(zip((key(r) for r in data["selected"]),
                                  profile.predict(data, weights[mode]), strict=True))
    near, far = SurfaceContrast(mount, near=True), SurfaceContrast(mount)
    lookup = tuple(np.concatenate((a, b[1:])) for a, b in zip(near.lookup["front"], far.lookup["front"], strict=True))
    manual = read(MANUAL)["frames"] if track == "ks_silverstone1967" else {}
    all_reports, rows, marks = {}, [], {}
    for run_id, folder in read(mapping).items():
        timelines = {name: [] for name in ("edge", "t4t5", *(f"{g}_{c}" for g in GEOMETRY[1:] for c in CONFIRM))}
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[run_id]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                k = run_id, item["capture_id"]
                if k not in truth:
                    continue
                t, reference = truth[k], old[k]
                name = Path(item["file"]).name
                assert archive.by_name[name]["sha256"] == item["file_sha256"]
                meta = archive.json(name)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, k[1], item["generation"])
                views = {v["view_id"]: v for v in item["views"]}
                usable = set(views) == {"front", "left", "right", "rear"} and not any(
                    src.get(flag) for src in (item, meta) for flag in ("aid_probe_only", "ideal_line_shown"))
                base = original_edges[k]
                geometry = dict.fromkeys(GEOMETRY, base)
                xy = None
                if usable and (not reference["baseline_observable"] or str(k[1]) in manual):
                    images = {side: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                                cv2.IMREAD_COLOR) for side, v in views.items()}
                    assert all(image is not None for image in images.values())
                    raw = near.detect(images) + far.detect({"front": images["front"]})[1:]
                    points = np.array([[r[f] if r[f] is not None else np.nan for f in FIELDS] for r in raw])
                    assert np.array_equal(np.where(np.array([r["valid"] for r in raw])[:, None], points, np.nan), base, equal_nan=True)
                    lab = {side: cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for side, image in images.items()}
                    evidence = np.r_[edge_scores(near, lab), edge_scores(far, lab)[1:]]
                    xy, normals = front_normals(lab["front"], lookup)
                    geometry["normal"] = choose(raw, evidence, near.lateral, "normal_fill")
                    bounded = evidence.copy()
                    # A grid endpoint has no measured evidence beyond the search interval.
                    bounded[:, [0, -1]] = -np.inf
                    geometry["bounded_normal"] = choose(raw, bounded, near.lateral, "normal_fill")
                annotation = manual.get(str(k[1]))
                if annotation and xy is not None:
                    mark = np.asarray(annotation["right_px"])
                    errors = {}
                    for g, edges in geometry.items():
                        ids = np.flatnonzero(np.isfinite(edges[:, 0]))
                        cells = abs(near.lateral[None, :] - edges[ids, 0, None]).argmin(axis=1)
                        visible = lookup[2][ids, cells]
                        errors[g] = distance_to_mark(xy[ids[visible], cells[visible]], mark) if len(mark) else []
                    marks[str(k[1])] = dict(kind=annotation["kind"], errors_px=errors)
                motion = read_motion(meta) if usable else None
                moving = motion is not None and motion[1] >= MIN_SPEED_MPS
                proposals, final = {}, {}
                for name, (_, old_mode) in CONFIRM.items():
                    final[name] = reference["proposals"][old_mode]
                for g, edges in geometry.items():
                    if g == "paired":
                        continue
                    local = decide(curvature(edges), .002) if moving else None
                    proposal = local if local and local[0] and 4. <= local[1] <= 20. else None
                    proposals[g] = proposal
                    for c, values in profiles.items():
                        evidence = values.get(k)
                        score = signals(evidence, *proposal)["local"] if proposal else None
                        final[f"{g}_{c}"] = final[c] or (proposal if score is not None and score >= .25 else None)
                for mode, values in timelines.items():
                    proposal = final[mode]
                    values.append(publish(t, proposal, reference["baseline_observable"] or proposal is not None))
                rows.append(dict(run=run_id, frame=k[1], truth=t["truth"], label=t["label"],
                                 sections={g: np.isfinite(x).all(axis=1).sum().item() for g, x in geometry.items()},
                                 proposals=proposals, final=final))
        all_reports[run_id] = {mode: summarize(values) for mode, values in timelines.items()}
    assert [key(r) for r in rows] == list(truth)
    names = next(iter(all_reports.values()))
    metrics = {m: totals({run_id: reports[m] for run_id, reports in all_reports.items()}) for m in names}
    path = OUT / f"{track}-frames.jsonl"
    jsonl(path, rows)
    return dict(metrics=metrics, manual=marks, rows_sha256=digest(path), bundles=bundles)


def main():
    cv2.setNumThreads(1)
    frozen = read(LOCK / "lock.json")
    model = LOCK / "confirmation-models.joblib"
    assert {Path(k).resolve(): v for k, v in frozen["hashes"].items()}[model.resolve()] == digest(model)
    weights = joblib.load(model)
    mount = Path("configs/ac-csp-exterior-front-v3-retina.json")
    hashes = {Path(__file__), Path(complete.__code__.co_filename), Path(edge_scores.__code__.co_filename),
              Path(curvature.__code__.co_filename), Path(summarize.__code__.co_filename), mount, model,
              LOCK / "lock.json", MANUAL, ORIGINAL / "result.json"}
    results = {}
    for track in ("ks_silverstone1967", "spa"):
        assert track not in frozen["training"]
        results[track] = run(track, read(mount), weights, hashes)
        print(track, results[track]["metrics"], flush=True)
    write(OUT / "result.json", dict(status="evaluated", seed=None, training=frozen["training"],
          posthoc=True, termination="all_recorded_frames_replayed", tracks=results,
          source_sha256={str(p): digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
