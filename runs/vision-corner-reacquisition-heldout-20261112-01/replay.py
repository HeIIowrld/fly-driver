"""Replay the fixed Silverstone warning-reacquisition rule across circuit folds."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from corner_fusion_boundaries import curvature, decide, read_motion, stations, transform
from evaluate_fused_corner import read_rows
from flybrain_drive.ac_camera_archive import CameraArchive
from image_normal import choose, edge_scores
from paired_motion import inputs
import profile_replay as profile
from relative_corner_boundaries import SurfaceContrast
from replay_learned import baseline_data, publish, signals
from score_corner_captures import MIN_SPEED_MPS, adjacent, summarize
from summarize_paired import category
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
RUNS = OUT.parent
FOLDS = RUNS / "vision-corner-local-heldout-20261112-01"
PRIOR = RUNS / "vision-corner-visibility-boundary-20261112-01"
MOUNT = Path("configs/ac-csp-exterior-front-v3-retina.json")
CONFIRM = {"edge": ("boundaries", "ranked_local_boundary_hold"),
           "t4t5": ("boundaries_temporal_t4t5", "ranked_local_t4t5_hold")}
CUTOFF = .25
CURVATURE = .002
TOLERANCE_M = 4.


def key(row):
    return row["run"], row["frame"]


def held_offer(row, mode):
    held = row["proposals"][mode]
    return held if row["proposals"][mode.removesuffix("_hold")] is None else None


def within(offer, previous, travel):
    remaining = previous[1] - travel
    return remaining > 0 and offer[0] == previous[0] and abs(offer[1] - remaining) <= TOLERANCE_M


def boundary(archive, views, near, far, measured, observable):
    if observable:
        return measured
    images = {name: cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name), np.uint8),
                                 cv2.IMREAD_COLOR) for name, view in views.items()}
    assert all(im is not None for im in images.values())
    raw = near.detect(images) + far.detect({"front": images["front"]})[1:]
    lab = {name: cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for name, im in images.items()}
    scores = np.r_[edge_scores(near, lab), edge_scores(far, lab)[1:]]
    scores[:, [0, -1]] = -np.inf
    assert np.array_equal(choose(raw, scores, near.lateral, "baseline"), measured, equal_nan=True)
    return choose(raw, scores, near.lateral, "normal_fill")


def metrics(timelines):
    reports = {run: summarize(rows) for run, rows in timelines.items()}
    totals = {name: sum(r["by_direction"][name]["direction_hits"] for r in reports.values())
              for name in ("left", "right")}
    outputs = Counter()
    for r in reports.values():
        outputs.update(r["outputs"])
    stable = {(e["physical_corner"], e["label"]) for r in reports.values() for e in r["events"]
              if e["first_stable_frame"] is not None}
    return dict(**totals, frames=sum(r["frames"] for r in reports.values()), outputs=dict(outputs),
                outside=sum(r["false_alerts"]["outside_band"]["frames"] for r in reports.values()),
                in_corner=sum(sum(r["truth_by_output"].get("in_corner", {}).get(s, 0) for s in ("left", "right"))
                              for r in reports.values()),
                joint=sum(s["joint_hits"] for r in reports.values() for s in r["by_direction"].values()),
                stable=len(stable), stable_right=sum(label == "2" for _, label in stable))


def run(track, hashes):
    fold = FOLDS / f"{track}-result.json"
    known = read(fold)
    frames = FOLDS / f"{track}-frames.jsonl"
    sections = FOLDS / f"{track}-sections.npz"
    model = FOLDS / f"{track}-fold-models.joblib"
    assert known["held_out"] == track and track not in known["training"]
    assert digest(frames) == known["replay"]["rows_sha256"]
    assert digest(sections) == known["replay"]["sections_sha256"]
    assert digest(model) == known["fold_models_sha256"]
    hashes.update((fold, frames, sections, model))
    old = {key(r):r for r in read_rows(frames)}
    with np.load(sections, allow_pickle=False) as arr:
        assert list(old) == list(zip(arr["run"], arr["frame"], strict=True))
        assert arr["modes"][0] == "baseline" and np.array_equal(arr["stations"], stations)
        measured = dict(zip(old, arr["sections"][:, 0], strict=True))
    group, _, mapping, bundles = baseline_data(track, hashes)
    truth = {key(r): r for r in group["rows"]}
    assert list(truth) == list(old)
    weights = joblib.load(model)
    input_groups = inputs(track, hashes)
    profiles = {}
    for name, (mode, _) in CONFIRM.items():
        data = input_groups[mode]
        assert list(truth) == [key(r) for r in data["rows"]]
        profiles[name] = dict(zip((key(r) for r in data["selected"]),
                                 profile.predict(data, weights[mode]), strict=True))
    near, far = SurfaceContrast(read(MOUNT), near=True), SurfaceContrast(read(MOUNT))
    timelines = {name: defaultdict(list) for mode in CONFIRM for name in (mode, mode+"_reacquired")}
    records, changed, paired = [], [], Counter()
    for tag, folder in read(mapping).items():
        previous = None
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[tag]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                k = tag, item["capture_id"]
                if k not in truth:
                    previous = None
                    continue
                t, source = truth[k], old[k]
                assert all(t[n] == source[n] for n in ("truth", "label", "physical_corner", "truth_endpoints_m"))
                filename = Path(item["file"]).name
                assert archive.by_name[filename]["sha256"] == item["file_sha256"]
                meta = archive.json(filename)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, k[1], item["generation"])
                views = {v["view_id"]: v for v in item["views"]}
                usable = set(views) == {"front", "left", "right", "rear"} and not any(
                    rec.get(flag) for rec in (item, meta) for flag in ("aid_probe_only", "ideal_line_shown"))
                motion = read_motion(meta) if usable else None
                old_alerts = {name: source["proposals"][hold] for name, (_, hold) in CONFIRM.items()}
                priors = {}
                travel = None
                if (motion is not None and motion[1] >= MIN_SPEED_MPS and previous is not None
                        and previous[2] is not None and adjacent(previous[0], t)):
                    step = transform(motion, previous[2])
                    if step is not None:
                        travel = float(np.linalg.norm(step[1]))
                        priors = {name: held_offer(previous[1], hold) for name, (_, hold) in CONFIRM.items()
                                  if old_alerts[name] is None}
                candidate, scores = None, {}
                if any(priors.values()):
                    edges = boundary(archive, views, near, far, measured[k], source["baseline_observable"])
                    proposed = decide(curvature(edges), CURVATURE)
                    if proposed and proposed[0] and 4. <= proposed[1] <= 20.:
                        candidate = proposed
                        scores = {name: signals(values.get(k), *candidate)["local"] for name, values in profiles.items()}
                final = dict(old_alerts)
                for name, prior in priors.items():
                    if candidate is not None and prior is not None and within(candidate, prior, travel):
                        score = scores[name]
                        if score is not None and score >= CUTOFF:
                            final[name] = candidate
                for name in CONFIRM:
                    for variant, proposal in ((name, old_alerts[name]), (name+"_reacquired", final[name])):
                        timelines[variant][tag].append(publish(t, proposal, source["baseline_observable"] or proposal is not None))
                record = dict(run=tag,frame=k[1],truth=t["truth"],label=t["label"],truth_endpoints_m=t["truth_endpoints_m"],
                              candidate=candidate,scores=scores,previous_held=priors,travel_m=travel,old=old_alerts,new=final)
                if old_alerts != final:
                    changed.append(record)
                if old_alerts["edge"] is None and old_alerts["t4t5"] is None and final["edge"] != final["t4t5"]:
                    for name, proposal in final.items():
                        if proposal is not None:
                            paired[name,category(source,proposal)] += 1
                records.append(record)
                previous = t, source, motion
    assert [key(r) for r in records] == list(truth)
    path = OUT / f"{track}-frames.jsonl"
    jsonl(path, records)
    checks = {name:metrics(rows) for name,rows in timelines.items()}
    for name, (_, hold) in CONFIRM.items():
        expected = known["counts"][hold]
        assert all(checks[name][k] == expected[k] for k in ("frames","left","right","joint","outside"))
        stable = {e["physical_corner"] for r in known["replay"]["methods"][hold].values()
                  for e in r["events"] if e["first_stable_frame"] is not None}
        assert checks[name]["stable"] == len(stable)
    return dict(training=known["training"],metrics=checks,changes=changed,rows_sha256=digest(path),
                paired_same_candidate_both_quiet=[dict(model=k[0],category=k[1],count=v) for k,v in sorted(paired.items())])


def main():
    cv2.setNumThreads(1)
    hashes={Path(__file__),MOUNT,PRIOR/"bridge-result.json",Path(choose.__code__.co_filename),
            Path(curvature.__code__.co_filename),Path(inputs.__code__.co_filename),Path(summarize.__code__.co_filename)}
    results={}
    for track in profile.CALIBRATION_TRACKS:
        results[track]=run(track,hashes)
        print(track,results[track]["metrics"],flush=True)
    write(OUT/"result.json",dict(status="evaluated",seed=None,posthoc_development=True,tracks=results,
          tolerance_m=TOLERANCE_M,confirmation_cut=CUTOFF,curvature_cut=CURVATURE,
          termination="all_circuit_folds_scored",source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
