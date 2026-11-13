"""Train a range readout from cross-fitted visual edge profiles and causal motion."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import hinge
import learn_edges
from corner_fusion_boundaries import curvature, stations
from evaluate_fused_corner import frame_key
from profile_replay import CALIBRATION_TRACKS, TRACKS
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import compare_tracks, score

OUT = Path(__file__).parent
MOTION = Path("runs/vision-corner-edge-agreement-20261111-01")
OFFSETS = np.arange(4., 22.)
SIGNS = np.array([1., -1.])
CUTS = (.50, .60, .70, .75, .80, .85, .90, .95, .975)
FIXED = .85
MODES = ("current", "aligned")


def geometry(edges):
    values = np.full(len(OFFSETS), np.nan)
    for distance, signed in curvature(edges):
        values[int(distance - OFFSETS[0])] = signed
    return values


def shape_features(rows, edges, travel):
    samples, ready = [], []
    last_row = previous = None
    for row, boundaries, step in zip(rows, edges, travel, strict=True):
        known = np.isfinite(boundaries).all(axis=1)
        current = geometry(boundaries)
        coverage = np.isfinite(current)
        usable = coverage.sum() >= 2
        # These statistics describe visible geometry only, including explicit observation masks.
        observed = stations[known]
        width = np.diff(boundaries[known], axis=1).ravel()
        summary = [known.mean(), coverage.mean(),
                   observed[0] / 24. if observed.size else 0.,
                   observed[-1] / 24. if observed.size else 0.,
                   float(np.mean(width)) / 10. if width.size else 0.,
                   float(np.std(width)) / 10. if width.size else 0.]
        base = np.concatenate((SIGNS[:, None] * np.nan_to_num(current)[None, :] * 100.,
                               np.broadcast_to(coverage, (2, len(OFFSETS))),
                               np.broadcast_to(summary, (2, len(summary)))), axis=1)
        overlap = np.zeros(len(OFFSETS), dtype=bool)
        aligned = np.zeros(len(OFFSETS))
        if (usable and previous is not None and adjacent(last_row, row) and 0. <= step <= 12.):
            earlier = np.interp(OFFSETS + step, OFFSETS, previous, left=np.nan, right=np.nan)
            overlap = coverage & np.isfinite(earlier)
            aligned[overlap] = earlier[overlap]
        history = np.concatenate((SIGNS[:, None] * aligned[None, :] * 100.,
                                  np.broadcast_to(overlap, (2, len(OFFSETS)))), axis=1)
        samples.append(np.column_stack((base, history)))
        ready.append(usable)
        last_row, previous = row, current if usable else None
    return np.stack(samples), np.asarray(ready)


def supervision(group, usable):
    rows = group["rows"]
    known = usable & np.array([r["truth"] in ("approach", "straight", "out_of_range", "in_corner")
                              for r in rows])
    targets = np.array([[r["truth"] == "approach" and r["label"] == side for side in ("1", "2")]
                        for r in rows])
    distance = np.array([r["truth_distance_m"] if r["truth"] == "approach" else 0. for r in rows])
    return known, targets, np.repeat(distance[:, None], 2, axis=1)


def learn_readout(parts, mode):
    features, labels, distances, weights = [], [], [], []
    for g, (shape, usable) in parts:
        known, target, distance = supervision(g, usable)
        x = select_mode(shape, mode)
        y = target[known].ravel()
        features.append(x[known].reshape(-1, x.shape[-1]))
        labels.append(y)
        distances.append(distance[known].ravel())
        weights.append(np.where(y, .5 / max(1, y.sum()), .5 / max(1, (~y).sum())))
    x, y, distance, w = map(np.concatenate, (features, labels, distances, weights))
    w *= len(w) / w.sum()
    scaler = StandardScaler().fit(x, sample_weight=w)
    transformed = scaler.transform(x)
    classifier = LogisticRegression(C=.08).fit(transformed, y, sample_weight=w)
    regressor = Ridge(alpha=70.).fit(transformed[y], distance[y], sample_weight=w[y])
    return scaler, classifier, regressor


def select_mode(shape, mode):
    if mode == "current":
        return shape[:, :, :2 * len(OFFSETS) + 6]
    return shape


def predict_readout(model, shape, mode):
    scaler, classifier, regressor = model
    x = select_mode(shape, mode)
    x = scaler.transform(x.reshape(-1, x.shape[-1]))
    p = classifier.predict_proba(x)[:, 1].reshape(-1, 2)
    d = regressor.predict(x).clip(4., 20.).reshape(-1, 2)
    return p, d


def output(group, usable, predictions, cutoff, mode):
    p, d = predictions
    for r, valid, scores, distances in zip(group["rows"], usable, p, d, strict=True):
        side = int(scores.argmax())
        if (not valid or scores[side] <= scores[1 - side] or scores[side] < cutoff):
            yield r
        elif mode == "append" and hinge.recoverable(r):
            yield dict(score(r, (side + 1, float(distances[side]))), abstain_reason=None)
        elif mode == "distance_only" and r["state"] == hinge.ALERTS[side]:
            yield score(r, (side + 1, float(distances[side])))
        else:
            yield r


def diagnostic(group, usable, predictions):
    known, y, truth = supervision(group, usable)
    p, d = predictions
    eligible = np.array([hinge.recoverable(r) for r in group["rows"]])
    measure = known & eligible
    positive = y & measure[:, None]
    return dict(known_eligible_frames=int(measure.sum()), eligible_positives=int(positive.sum()),
                eligible_positive_rate=float(y[measure].mean()),
                eligible_average_precision=float(average_precision_score(y[measure].ravel(), p[measure].ravel())),
                true_side_distance_mae_m=float(np.abs(d - truth)[positive].mean()))


def run():
    hashes = {Path(__file__), learn_edges.OUT / "learn_edges.py", hinge.OUT / "hinge.py",
              Path("scripts/corner_fusion_boundaries.py"), Path("scripts/score_corner_captures.py"),
              Path("scripts/evaluate_independent_corner.py"),
              Path("runs/vision-corner-onset-support-20261111-01/transfer.py")}
    groups, data, travel, examples = {}, {}, {}, {}
    for track in TRACKS:
        g, raw = learn_edges.read_inputs(track, hashes)
        motion = MOTION / f"{track}-motion.npz"
        proof = motion.with_suffix(".json")
        if digest(motion) != read(proof)["motion_sha256"]:
            raise ValueError(f"Relative-motion arrays changed: {track}")
        with np.load(motion) as m:
            if list(zip(m["run"], m["frame"], strict=True)) != [frame_key(r) for r in g["rows"]]:
                raise ValueError(f"Motion frame order differs: {track}")
            travel[track] = m["motion"]
        hashes.update((motion, proof))
        groups[track], data[track] = g, raw
        if track in CALIBRATION_TRACKS:
            examples[track] = learn_edges.examples(raw)
    print("loaded all six track inputs", flush=True)

    @lru_cache(maxsize=None)
    def train_patch(train):
        if "spa" in train or len(train) < 3:
            raise ValueError("Patch fitting needs three or more development tracks")
        return learn_edges.train([examples[name] for name in train])

    @lru_cache(maxsize=None)
    def geometry_for(track, train):
        if track in train:
            raise ValueError("Patch-model evaluation overlaps training")
        score_grid = learn_edges.patch_scores(train_patch(train), data[track])
        edges = np.stack([learn_edges.sections(scores) for scores in score_grid])
        print("geometry", track, train, flush=True)
        return shape_features(groups[track]["rows"], edges, travel[track])

    result, diagnostic_rows = {}, []
    for track in TRACKS:
        train = tuple(t for t in CALIBRATION_TRACKS if t != track)
        g = groups[track]
        train_geometry = {key:geometry_for(key, tuple(t for t in train if t != key)) for key in train}
        shape, usable = geometry_for(track, train)
        comparisons = {}
        for mode in MODES:
            model = learn_readout([(groups[key], x) for key, x in train_geometry.items()], mode)
            predictions = predict_readout(model, shape, mode)
            reports = {}
            for change in ("append", "distance_only"):
                sweep = {}
                for cutoff in CUTS:
                    rows = list(output(g, usable, predictions, cutoff, change))
                    sweep[str(cutoff)] = compare_tracks(g["original"], rows, g["rules"])
                reports[change] = dict(fixed=sweep[str(FIXED)], sweep=sweep)
            comparisons[mode] = dict(diagnostic=diagnostic(g, usable, predictions), policies=reports)
            for row, okay, p, d in zip(g["rows"], usable, *predictions, strict=True):
                if row["truth"] == "approach" or (okay and p.max() >= FIXED and hinge.recoverable(row)):
                    diagnostic_rows.append(dict(track=track, run=row["run"], frame=row["frame"],
                        mode=mode, usable=bool(okay), truth=row["truth"], label=row["label"],
                        truth_distance_m=row["truth_distance_m"], near_scores=p.tolist(), distance_m=d.tolist()))
        result[track] = dict(patch_training_for_test=train,
                             patch_training_for_readout={key:tuple(t for t in train if t != key) for key in train},
                             baseline=g["baseline"], comparisons=comparisons)
        print(track, {m:(r["diagnostic"], r["policies"]["append"]["fixed"]["metrics"]["candidate"])
                     for m, r in comparisons.items()}, flush=True)
    path = OUT / "rows.jsonl"
    jsonl(path, diagnostic_rows)
    write(OUT / "result.json", dict(tracks=result, rows_sha256=digest(path),
          parameters=dict(fixed_cutoff=FIXED, cutoffs=CUTS, seed=learn_edges.SEED),
          source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="cross_fitted_visual_shape_readout_complete"))


if __name__ == "__main__":
    run()
