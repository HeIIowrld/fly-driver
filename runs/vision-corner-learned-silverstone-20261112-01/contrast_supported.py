"""Require both projected patch probability and image-normal contrast for replacements."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np

import replay_learned as runner
from train_vision_teacher_line import digest, read, write

ROOT = Path(__file__).parent
OUT = ROOT / "contrast-supported"


def ranked_patches(raw, contrast, grid, probability):
    scores = contrast.copy()
    for i, row in enumerate(raw):
        right, left = (row[key] for key in runner.FIELDS)
        for side, known, opposite, sign in ((0, right, left, -1.), (1, left, right, 1.)):
            if known is not None or opposite is None:
                continue
            width = (grid-opposite)*sign
            region = (width > 3.) & (width < 20.)
            score = np.interp(grid, runner.GRID, probability[i, side], left=0., right=0.)
            scores[i, region] = np.where(np.isfinite(contrast[i, region]) &
                                        (score[region] >= runner.learned.PATCH_THRESHOLD),
                                        score[region], -np.inf)
    return runner.choose(raw, scores, grid, "normal_fill")


def main():
    cv2.setNumThreads(1)
    freeze = ROOT/"frozen.json"
    source = read(freeze)
    checkpoint = ROOT/"patch-model.joblib"
    assert digest(checkpoint) == source["checkpoint_sha256"]
    assert set(source["training"]).isdisjoint(runner.TEST_TRACKS)
    for filename, sha in source["source_sha256"].items():
        assert digest(Path(filename)) == sha, filename
    OUT.mkdir(exist_ok=False)
    hashes = {Path(__file__), freeze, checkpoint}
    model = joblib.load(checkpoint)
    profiles = runner.confirmation_models(tuple(source["training"]), hashes)
    config = Path(runner.load_frozen(runner.PROTOCOL.with_name("candidate.json"))["camera_config"])
    hashes.add(config)
    runner.OUT = OUT
    runner.ranked_patches = ranked_patches
    mount = read(config)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {t: pool.submit(runner.replay, t, mount, model, profiles[t]) for t in runner.TEST_TRACKS}
        tracks = {t: f.result() for t, f in futures.items()}
    write(OUT/"result.json", dict(status="evaluated", tracks=tracks, seed=source["seed"],
          termination="archive_end", source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
