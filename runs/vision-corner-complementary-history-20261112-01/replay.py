"""Compare independent-side history with the same current-frame boundary decoder."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from corner_complementary_history import EdgeHistory
from corner_fusion_boundaries import curvature, decide, read_motion, transform
from evaluate_fused_corner import read_rows
from flybrain_drive.ac_camera_archive import CameraArchive
from image_normal import FIELDS, choose, edge_scores
from paired_motion import inputs
import profile_replay as profile
from relative_corner_boundaries import SurfaceContrast
from replay_learned import baseline_data, publish, signals
from score_corner_captures import MIN_SPEED_MPS, adjacent, summarize
from summarize_paired import category
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
FOLDS = OUT.parent / "vision-corner-local-heldout-20261112-01"
OLD = OUT.parent / "vision-corner-local-curvature-20261112-01"
LOCK = OUT.parent / "vision-vallelunga-local-corner-protocol-20261112-01"
MOUNT = Path("configs/ac-csp-exterior-front-v3-retina.json")
MODELS = {"edge": ("boundaries", "ranked_local_boundary_hold"),
          "t4t5": ("boundaries_temporal_t4t5", "ranked_local_t4t5_hold")}
MODES = ("old", "current", "history_1", "history_2")
CHECK = ("ks_silverstone1967", "spa")


def key(row):
    return row["run"], row["frame"]


def totals(timeline):
    reports = [summarize(rows) for rows in timeline.values()]
    hits = Counter()
    for r in reports:
        for side, scores in r["by_direction"].items():
            hits[side] += scores["direction_hits"]
            hits["joint"] += scores["joint_hits"]
        hits["outside"] += r["false_alerts"]["outside_band"]["frames"]
        hits["in_corner"] += sum(r["truth_by_output"].get("in_corner", {}).get(s,0) for s in ("left","right"))
        hits["frames"] += r["frames"]
        hits["no_output"] += r["outputs"]["no_output"]
    hits["stable"] = len({e["physical_corner"] for r in reports for e in r["events"]
                          if e["first_stable_frame"] is not None})
    return dict(hits)


def setup(track, hashes):
    group, _, mapping, bundles = baseline_data(track, hashes)
    if track in CHECK:
        model = LOCK/"confirmation-models.joblib"
        lock = LOCK/"lock.json"
        frozen = read(lock)
        training = frozen["training"]
        assert {Path(k).resolve():v for k,v in frozen["hashes"].items()}[model.resolve()]==digest(model)
        source = OLD/f"{track}-frames.jsonl"
        receipt = OLD/"result.json"
        proof = read(receipt)["tracks"][track]
    else:
        model = FOLDS/f"{track}-fold-models.joblib"
        source = FOLDS/f"{track}-frames.jsonl"
        receipt = FOLDS/f"{track}-result.json"
        fold = read(receipt)
        training = fold["training"]
        assert digest(model)==fold["fold_models_sha256"] and fold["held_out"]==track
        proof = fold["replay"]
    assert track not in training and digest(source)==proof["rows_sha256"]
    hashes.update((model,source,receipt))
    if track in CHECK:
        hashes.add(lock)
    rows = read_rows(source)
    assert [key(r) for r in rows]==[key(r) for r in group["rows"]]
    data = inputs(track,hashes)
    weights = joblib.load(model)
    profiles = {}
    for name, (mode, _) in MODELS.items():
        features = data[mode]
        assert [key(r) for r in features["rows"]]==[key(r) for r in rows]
        profiles[name] = dict(zip((key(r) for r in features["selected"]),
                                   profile.predict(features,weights[mode]),strict=True))
    return group["rows"], rows, mapping, bundles, profiles, training


def read_edges(archive, views, near, far):
    images = {name: cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name), np.uint8),
                                 cv2.IMREAD_COLOR) for name,view in views.items()}
    assert all(im is not None for im in images.values())
    raw = near.detect(images) + far.detect({"front":images["front"]})[1:]
    points = np.asarray([[r[n] if r[n] is not None else np.nan for n in FIELDS] for r in raw])
    lab = {name:cv2.cvtColor(im,cv2.COLOR_BGR2LAB) for name,im in images.items()}
    evidence = np.r_[edge_scores(near,lab),edge_scores(far,lab)[1:]]
    evidence[:,[0,-1]] = -np.inf
    filled = choose(raw,evidence,near.lateral,"normal_fill")
    # A newly guessed pair with neither measured side is not an observed anchor.
    return np.where(np.isfinite(filled).all(axis=1)[:,None] & np.isfinite(points).any(axis=1)[:,None],
                    filled,points)


def offer(edges):
    candidate = decide(curvature(edges),.002)
    return candidate if candidate and candidate[0] and 4. <= candidate[1] <= 20. else None


def run(track, hashes):
    rows, old, mapping, bundles, profiles, training = setup(track,hashes)
    truth = {key(r):r for r in rows}
    old = {key(r):r for r in old}
    near,far = (SurfaceContrast(read(MOUNT),near=value) for value in (True,False))
    timelines = {f"{m}_{g}":defaultdict(list) for m in MODELS for g in MODES}
    records, comparison = [], Counter()
    for tag,folder in read(mapping).items():
        history, previous = EdgeHistory(), None
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                k = tag,item["capture_id"]
                if k not in truth:
                    previous = None
                    continue
                t, reference = truth[k],old[k]
                name = Path(item["file"]).name
                assert archive.by_name[name]["sha256"] == item["file_sha256"]
                meta = archive.json(name)
                assert (meta["track_id"],meta["index"],meta["generation"]) == (track,k[1],item["generation"])
                views = {v["view_id"]:v for v in item["views"]}
                usable = set(views)=={"front","left","right","rear"} and not any(
                    rec.get(flag) for rec in (meta,item) for flag in ("aid_probe_only","ideal_line_shown"))
                motion = read_motion(meta) if usable else None
                moving = motion is not None and motion[1]>=MIN_SPEED_MPS
                step = transform(motion,previous[1]) if moving and previous and adjacent(previous[0],t) else None
                base = {m:reference["proposals"][v[1]] for m,v in MODELS.items()}
                proposals, counts = {}, {}
                if moving:
                    current = read_edges(archive,views,near,far)
                    fused, support = history.update(motion[0],current,step)
                    geometry = {"current":current,**{f"history_{n}":e for n,e in fused.items()}}
                    proposals = {g:offer(e) for g,e in geometry.items()}
                    counts = {g:int(np.isfinite(e).all(axis=1).sum()) for g,e in geometry.items()}
                final = {}
                for model in MODELS:
                    values = profiles[model].get(k)
                    for g in MODES:
                        candidate = proposals.get(g)
                        score = signals(values,*candidate)["local"] if candidate else None
                        decision = base[model] or (candidate if score is not None and score>=.25 else None)
                        mode = f"{model}_{g}"
                        final[mode] = decision
                        timelines[mode][tag].append(publish(t,decision,reference["baseline_observable"] or decision is not None))
                for g in ("history_1","history_2"):
                    if proposals.get(g) != proposals.get("current"):
                        comparison[g,category(t,proposals.get("current")),category(t,proposals.get(g))] += 1
                records.append(dict(run=tag,frame=k[1],truth=t["truth"],label=t["label"],
                                    truth_endpoints_m=t["truth_endpoints_m"],sections=counts,
                                    proposals=proposals,final=final))
                previous = (t,motion) if moving else None
    assert [key(r) for r in records]==list(truth)
    path = OUT/f"{track}-frames.jsonl"
    jsonl(path,records)
    return dict(training=training,rows_sha256=digest(path),
                metrics={m:totals(t) for m,t in timelines.items()},
                proposal_transitions=[dict(mode=k[0],before=k[1],after=k[2],frames=n)
                                      for k,n in sorted(comparison.items())])


def main():
    cv2.setNumThreads(1)
    hashes = {Path(__file__), MOUNT, Path("scripts/corner_complementary_history.py"),
              Path(edge_scores.__code__.co_filename), Path(curvature.__code__.co_filename),
              Path(summarize.__code__.co_filename)}
    results = {}
    for track in (*CHECK,*profile.CALIBRATION_TRACKS):
        results[track] = run(track,hashes)
        print(track,results[track]["metrics"],flush=True)
        write(OUT/"progress.json",dict(complete=list(results)))
    write(OUT/"result.json",dict(status="evaluated",seed=None,posthoc=True,tracks=results,
          age_s=.24,within_history_spread_m=.8,termination="all_timelines_scored",
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
