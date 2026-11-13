"""Replay the new patch model through the unchanged local-curvature and T4/T5 comparison."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from replay_hard_negatives import OUT, TRAIN, cv2, digest, joblib, read, replay, revision, write

import replay_local as local


def main():
    cv2.setNumThreads(1)
    local.checks()
    checkpoint = OUT / "hard-negatives.joblib"
    control = local.OUT / "result.json"
    old = read(control)
    local.OUT = OUT / "local"
    local.OUT.mkdir(exist_ok=True)
    config = Path(replay.load_frozen(replay.PROTOCOL.with_name("candidate.json"))["camera_config"])
    sources = {Path(__file__), control, config, checkpoint, Path(local.__file__)}
    profiles = replay.confirmation_models(TRAIN, sources)
    model = joblib.load(checkpoint)
    mount = read(config)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {t: executor.submit(local.replay, t, mount, model, profiles[t])
                for t in replay.TEST_TRACKS}
        tracks = {t: job.result() for t, job in jobs.items()}
    for track, result in tracks.items():
        previous = old["tracks"][track]["methods"]
        for mode in ("baseline", "normal_local"):
            if result["methods"][mode] != previous[mode]:
                raise ValueError(f"Unchanged control did not reproduce: {track}, {mode}")
    write(local.OUT / "result.json", dict(
        seed=0, source_revision=revision(), training=TRAIN, excluded=replay.TEST_TRACKS,
        tracks=tracks, source_sha256={str(p): digest(p) for p in sorted(sources)},
        curvature_cut=local.CUT, confirmation_cut=local.CONFIRM_THRESHOLD,
        termination="local_curvature_replays_and_control_parity_complete",
    ))


if __name__ == "__main__":
    main()
