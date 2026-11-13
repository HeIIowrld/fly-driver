"""Compare random negative sampling with hard alternatives on the same road section."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
for name in (
    "src", "scripts",
    "runs/vision-next-corner-deps-20261110",
    "runs/vision-corner-learned-edge-20261111-01",
    "runs/vision-corner-learned-silverstone-20261112-01",
    "runs/vision-silverstone-boundary-baseline-20261112-01",
    "runs/vision-corner-causal-edge-recovery-20261112-01",
    "runs/vision-corner-image-normal-20261112-01",
    "runs/vision-corner-connected-boundary-20261111-01",
    "runs/vision-corner-boundary-hinge-20261111-01",
    "runs/vision-corner-edge-reference-20261111-01",
    "runs/vision-corner-onset-support-20261111-01",
    "runs/vision-corner-curvature-profile-20261111-01",
    "runs/vision-corner-motion-confirmation-20261112-01",
    "runs/vision-corner-matched-approaches-20261111-01",
    "runs/vision-corner-joint-approach-20261111-01",
    "runs/vision-corner-head-ablation-20261112-01",
    "runs/vision-corner-motion-anchored-distance-20261111-01",
    "runs/vision-corner-luminance-boundary-20261112-01",
    "runs/vision-corner-motion-transfer-20261112-01",
    "runs/vision-corner-local-curvature-20261112-01",
    "runs/vision-corner-fused-confirmation-20261111-01",
):
    sys.path.append(str(ROOT / name))

import cv2
import joblib
import numpy as np

import learn_edges as learned
import replay_learned as replay
from patches import stations
from train_vision_teacher_line import digest, read, write

HARD_POOL = 6
HARD_COUNT = 2
TRAIN = tuple(replay.profile.CALIBRATION_TRACKS)


def load(track, sources):
    path = learned.OUT / f"{track}-patches.npz"
    receipt = path.with_suffix(".json")
    proof = read(receipt)
    if proof["track"] != track or digest(path) != proof["patches_sha256"]:
        raise ValueError(f"Changed arrays for {track}")
    for filename, sha in proof["input_sha256"].items():
        source = Path(filename)
        if source not in sources and digest(source) != sha:
            raise ValueError(f"Changed patch source: {source}")
        sources.add(source)
    sources.update((path, receipt))
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("features", "labels", "usable", "valid")}


def hard_examples(data, seed_model):
    rng = np.random.default_rng(learned.SEED)
    labels = data["labels"].reshape(-1, data["labels"].shape[-1])
    features = data["features"].reshape(*labels.shape, -1)
    positive = np.flatnonzero(labels.ravel() == 1)
    count = min(len(positive), learned.MAX_POSITIVES)
    chosen = rng.choice(positive, count, replace=False)
    group, lateral = np.unravel_index(chosen, labels.shape)
    negatives = np.flatnonzero(labels.ravel() == 0)
    easy = rng.choice(negatives, count * 4, replace=False).reshape(count, 4)
    negative_features = features.reshape(labels.size, -1)[easy].copy()
    pool = []
    destinations = []
    for index, group_index in enumerate(group):
        alternatives = np.flatnonzero(labels[group_index] == 0)
        if len(alternatives) >= HARD_COUNT:
            sampled = rng.choice(alternatives, min(len(alternatives), HARD_POOL), replace=False)
            sampled = np.pad(sampled, (0, HARD_POOL - len(sampled)), constant_values=sampled[0])
            pool.append(sampled)
            destinations.append(index)
    candidates = features[group[destinations, None], np.array(pool)]
    scores = seed_model.predict_proba(learned.encode(candidates))[:, 1].reshape(-1, HARD_POOL)
    # Repeated columns only pad short rows and cannot be selected twice.
    indices = np.array(pool)
    duplicates = np.tril(indices[:, :, None] == indices[:, None, :], k=-1).any(axis=-1)
    scores[duplicates] = -np.inf
    best = np.argsort(-scores, axis=1, kind="stable")[:, :HARD_COUNT]
    negative_features[destinations, :HARD_COUNT] = candidates[np.arange(len(pool))[:, None], best]
    x = np.concatenate((features[group, lateral], negative_features.reshape(count * 4, -1)))
    y = np.r_[np.ones(count, dtype=np.int8), np.zeros(count * 4, dtype=np.int8)]
    weights = np.where(y == 1, .5 / count, .5 / (4 * count))
    return (learned.encode(x), y, weights), len(destinations) * HARD_COUNT


def patch_report(data, scores):
    labels = data["labels"]
    positives = labels == 1
    negatives = labels == 0
    # Include every valid candidate at evaluation. Unlabeled alternatives are
    # counted separately since the projected reference cannot adjudicate them.
    valid = data["valid"][None] & data["usable"][:, None, None, None]
    winners = np.argmax(np.where(valid, scores, -np.inf), axis=-1, keepdims=True)
    winning_label = np.take_along_axis(labels, winners, axis=-1)[..., 0]
    positive_score = np.where(positives, scores, -np.inf).max(axis=-1)
    worst_rank = (negatives & (scores >= positive_score[..., None])).sum(axis=-1)
    known = positives.any(axis=-1) & negatives.any(axis=-1)
    summaries = {}
    for name, station_filter in (
        ("all", np.ones(len(stations), dtype=bool)),
        ("near", stations <= 10.),
        ("far", stations > 10.),
    ):
        for side_name, sides in (("both", slice(None)), ("right", 0), ("left", 1)):
            take = np.s_[:, station_filter, sides]
            mask = known[take]
            winning = winning_label[take][mask]
            ranks = worst_rank[take][mask]
            accepted = positive_score[take][mask] >= learned.PATCH_THRESHOLD
            summaries[f"{name}_{side_name}"] = dict(
                sections=int(mask.sum()), top_labeled=int((ranks == 0).sum()),
                top_positive=int((winning == 1).sum()), top_negative=int((winning == 0).sum()),
                top_unlabeled=int((winning == -1).sum()), accepted_positive=int(accepted.sum()),
                accepted_top_labeled=int((accepted & (ranks == 0)).sum()),
            )
    return summaries


def train_models(training, examples, arrays):
    base = learned.train([examples[t] for t in training])
    samples = [hard_examples(arrays[t], base) for t in training]
    return base, learned.train([sample for sample, _ in samples]), dict(
        zip(training, [count for _, count in samples], strict=True)
    )


def revision():
    safe = Path.cwd().resolve().as_posix()
    return subprocess.check_output(["git", "-c", f"safe.directory={safe}", "rev-parse", "HEAD"], text=True).strip()


def main():
    cv2.setNumThreads(1)
    sources = {Path(__file__), Path(learned.__file__), Path(replay.__file__)}
    arrays = {t: load(t, sources) for t in (*TRAIN, "spa")}
    examples = {t: learned.examples(arrays[t]) for t in TRAIN}
    folds = {}
    for track, data in arrays.items():
        training = tuple(t for t in TRAIN if t != track)
        base, hard, counts = train_models(training, examples, arrays)
        folds[track] = dict(training=training, hard_negative_counts=counts,
                            random=patch_report(data, learned.patch_scores(base, data)),
                            hard=patch_report(data, learned.patch_scores(hard, data)))
        print(track, "patch fold complete", flush=True)
    write(OUT / "patch-results.json", folds)

    assert set(TRAIN).isdisjoint(replay.TEST_TRACKS)
    _, hard, counts = train_models(TRAIN, examples, arrays)
    checkpoint = OUT / "hard-negatives.joblib"
    joblib.dump(hard, checkpoint)
    prior = replay.OUT / "result.json"
    prior_results = read(prior)
    sources.add(prior)
    replay.OUT = OUT
    config = Path(replay.load_frozen(replay.PROTOCOL.with_name("candidate.json"))["camera_config"])
    sources.add(config)
    profiles = replay.confirmation_models(TRAIN, sources)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {t: executor.submit(replay.replay, t, read(config), hard, profiles[t])
                   for t in replay.TEST_TRACKS}
        tracks = {t: future.result() for t, future in futures.items()}
    for track, result in tracks.items():
        previous = prior_results["tracks"][track]["methods"]
        for method in ("original", "normal"):
            if result["methods"][method] != previous[method]:
                raise ValueError(f"Unchanged control did not reproduce: {track}, {method}")
    write(OUT / "result.json", dict(
        seed=learned.SEED, source_revision=revision(), training=TRAIN, excluded=replay.TEST_TRACKS,
        patch_threshold=learned.PATCH_THRESHOLD, confirmation_threshold=replay.CONFIRM_THRESHOLD,
        alternatives_per_positive=HARD_COUNT, sampled_pool=HARD_POOL, hard_negative_counts=counts,
        checkpoint_sha256=digest(checkpoint), patch_results_sha256=digest(OUT / "patch-results.json"),
        source_sha256={str(p): digest(p) for p in sorted(sources)},
        tracks=tracks, termination="patch_folds_and_saved_video_replays_complete",
    ))


if __name__ == "__main__":
    main()
