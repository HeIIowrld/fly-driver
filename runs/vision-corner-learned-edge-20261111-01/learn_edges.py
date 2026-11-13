"""Train a patch boundary score without the evaluated circuit and replay its warnings."""
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from connected import connect, pair_states, retain
from evaluate_fused_corner import frame_key
import hinge
from hinge import CUTS, evaluate, evidence_summary, fit, group
from patches import GRID, SIDES, stations
from profile_replay import CALIBRATION_TRACKS, TRACKS
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
PROPOSAL_THRESHOLD = .04
PATCH_THRESHOLD = .55
SMOOTHNESS = .60
SEED = 0
MAX_POSITIVES = 7500


def encode(values):
    patch = values.astype(np.float32).reshape(-1, 5, 3)
    inward = patch[:, :2].mean(axis=1)
    outward = patch[:, 3:].mean(axis=1)
    contrast = outward - inward
    transition = patch[:, 3] - patch[:, 1]
    variation = np.abs(patch[:, 0] - patch[:, 1]) + np.abs(patch[:, 4] - patch[:, 3])
    return np.column_stack((patch.reshape(-1, 15), contrast, transition, variation))


def read_inputs(track, hashes):
    path = OUT / f"{track}-patches.npz"
    proof = path.with_suffix(".json")
    receipt = read(proof)
    if digest(path) != receipt["patches_sha256"]:
        raise ValueError(f"Changed patch arrays: {track}")
    for filename, sha in receipt["input_sha256"].items():
        source = Path(filename)
        if digest(source) != sha:
            raise ValueError(f"Changed patch source: {source}")
        hashes.add(source)
    hashes.update((path, proof))
    g = group(track, hashes)
    with np.load(path) as data:
        if list(zip(data["run"], data["frame"], strict=True)) != [frame_key(r) for r in g["rows"]]:
            raise ValueError(f"Frame order differs for {track}")
        return g, {key:data[key] for key in ("features", "labels", "reference", "usable", "valid")}


def examples(data):
    rng = np.random.default_rng(SEED)
    y = data["labels"].ravel()
    x = data["features"].reshape(len(y), -1)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    count = min(len(positive), MAX_POSITIVES)
    if not count or len(negative) < count * 4:
        raise ValueError("Patch labels lack enough positives or negatives")
    indices = np.r_[rng.choice(positive, count, replace=False),
                    rng.choice(negative, count * 4, replace=False)]
    weights = np.where(y[indices] == 1, .5 / count, .5 / (count * 4))
    return encode(x[indices]), y[indices], weights


def train(samples):
    x, y, weights = map(np.concatenate, zip(*samples, strict=True))
    model = HistGradientBoostingClassifier(max_iter=65, max_leaf_nodes=15, min_samples_leaf=40,
                                           l2_regularization=1., early_stopping=False, random_state=SEED)
    return model.fit(x, y, sample_weight=weights * (len(y) / weights.sum()))


def patch_scores(model, data):
    usable = data["usable"][:, None, None, None] & data["valid"][None]
    output = np.zeros(usable.shape, dtype=np.float32)
    for i in range(0, len(usable), 64):
        sl = slice(i, i + 64)
        valid = usable[sl]
        if valid.any():
            probabilities = model.predict_proba(encode(data["features"][sl][valid]))[:, 1]
            output[sl][valid] = probabilities
    return output


def sections(scores):
    groups = []
    for distance, layer in zip(stations, scores, strict=True):
        sides = []
        for side, values in zip(SIDES, layer, strict=True):
            eligible = values >= PATCH_THRESHOLD
            if distance <= 4.:
                eligible &= side * GRID >= 1.3
            strengths = np.where(eligible, 14. + 30. * (values - .5), -np.inf)
            sides.append(retain(GRID, strengths, side))
        groups.append(pair_states(*sides))
    return connect(stations, groups, SMOOTHNESS)


def edge_diagnostic(group, data, scores, learned, original):
    labels = data["labels"]
    valid_labels = labels >= 0
    raw, new = np.isfinite(original), np.isfinite(learned)
    reference = data["reference"]
    joint = raw & new & np.isfinite(reference)
    new_only = new & np.isfinite(reference)
    return dict(
        labeled_positives=int((labels == 1).sum()), labeled_negatives=int((labels == 0).sum()),
        patch_average_precision=float(average_precision_score(labels[valid_labels], scores[valid_labels])),
        patch_positive_rate=float(labels[valid_labels].mean()),
        input_complete_frames=int(data["usable"].sum()),
        original_pair_sections=int(raw.all(axis=-1).sum()), learned_pair_sections=int(new.all(axis=-1).sum()),
        joint_edge_points=int(joint.sum()),
        original_lateral_mae_m=float(np.abs(original - reference)[joint].mean()) if joint.any() else None,
        learned_lateral_mae_m=float(np.abs(learned - reference)[joint].mean()) if joint.any() else None,
        learned_all_observed_lateral_mae_m=float(np.abs(learned - reference)[new_only].mean()) if new_only.any() else None,
    )


def run():
    hashes = {Path(__file__), hinge.OUT / "hinge.py", Path("scripts/score_corner_captures.py"),
              Path("scripts/evaluate_independent_corner.py"),
              Path("runs/vision-corner-onset-support-20261111-01/transfer.py"),
              Path("runs/vision-corner-connected-boundary-20261111-01/connected.py")}
    groups, data, training = {}, {}, {}
    for track in TRACKS:
        groups[track], data[track] = read_inputs(track, hashes)
        if track in CALIBRATION_TRACKS:
            training[track] = examples(data[track])
        print("loaded", track, flush=True)
    result, details = {}, []
    for track in TRACKS:
        others = tuple(t for t in CALIBRATION_TRACKS if t != track)
        g = groups[track]
        probabilities = patch_scores(train([training[t] for t in others]), data[track])
        learned = np.stack([sections(scores) for scores in probabilities])
        with np.load(hinge.OUT / f"{track}-sections.npz") as old:
            original = old["values"][:, 0]
        g["fits"]["learned"] = [fit(stations, arr) for arr in learned]
        diagnostics = edge_diagnostic(g, data[track], probabilities, learned, original)
        before = g["baseline"]
        coverage = {m:evidence_summary(g, m) for m in ("current", "learned")}
        comparisons = {m:dict(fixed=evaluate(g, "learned", m, PROPOSAL_THRESHOLD),
                              sweep={str(c):evaluate(g, "learned", m, c) for c in CUTS})
                       for m in ("append", "confirm")}
        for row, (_, proposal) in zip(g["rows"], g["fits"]["learned"], strict=True):
            if row["truth"] == "approach" or proposal is not None:
                details.append(dict(track=track, run=row["run"], frame=row["frame"], truth=row["truth"],
                                    label=row["label"], truth_distance_m=row["truth_distance_m"],
                                    physical_corner=row["physical_corner"], proposal=proposal))
        result[track] = dict(training_circuits=others, diagnostics=diagnostics,
                             baseline=before, coverage=coverage, comparisons=comparisons)
        print(track, diagnostics, comparisons["append"]["fixed"]["metrics"]["candidate"], flush=True)
    path = OUT / "rows.jsonl"
    jsonl(path, details)
    write(OUT / "result.json", dict(tracks=result, rows_sha256=digest(path),
          parameters=dict(seed=SEED, proposal_threshold=PROPOSAL_THRESHOLD, patch_threshold=PATCH_THRESHOLD,
                          smoothness=SMOOTHNESS, max_positive_patches_per_track=MAX_POSITIVES),
          source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="learned_boundary_excluded_circuit_replays_complete"))


if __name__ == "__main__":
    run()
