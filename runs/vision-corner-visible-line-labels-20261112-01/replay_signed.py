"""Compare signed luminance with the fixed absolute contrast replay."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "vision-corner-image-row-trace-20261112-01"
sys.path.insert(0, str(OLD))

import replay_supported as supported

replay = supported.old


def signed_evidence(image, side):
    lab = replay.cv2.cvtColor(image, replay.cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = replay.cv2.blur(lab, (3, 1))
    scores = []
    for dx, dy in ((3 * side, 0), (2 * side, -2), (side, -3)):
        inside = np.roll(lab, (dy, dx), axis=(0, 1))
        outside = np.roll(lab, (-dy, -dx), axis=(0, 1))
        difference = outside - inside
        chroma = np.linalg.norm(difference[:, :, 1:], axis=2)
        inner_chroma = np.linalg.norm(inside[:, :, 1:] - 128., axis=2)
        outer_chroma = np.linalg.norm(outside[:, :, 1:] - 128., axis=2)
        scores.append(chroma - .35 * difference[:, :, 0]
                      + .25 * np.clip(outer_chroma - inner_chroma, -8., 16.))
    evidence = np.maximum.reduce(scores)
    evidence[:5] = evidence[-5:] = 0
    evidence[:, :5] = evidence[:, -5:] = 0
    return evidence


def check_polarity():
    image = np.full((180, 320, 3), 130, dtype=np.uint8)
    image[95:101, 20:300] = 40
    for side in (-1, 1):
        standard = supported.directional_evidence(image, side)
        signed = signed_evidence(image, side)
        assert standard[94, 220] > 15 and standard[101, 220] > 15
        assert signed[94, 220] < 1 and signed[101, 220] > 15


def main():
    check_polarity()
    replay.cv2.setNumThreads(1)
    supported.directional_evidence = signed_evidence
    replay.OUT = OUT
    replay.FrontTrace = supported.SupportedTrace
    entry = replay.context()
    runner = entry.import_replay()
    mount = replay.read(replay.MOUNT)
    tracks = runner.CHECK
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {t: executor.submit(replay.replay_track, t, entry, runner, mount) for t in tracks}
        results = {}
        for track, job in jobs.items():
            results[track] = job.result()
            print(track, results[track]["metrics"], flush=True)
    sources = {Path(__file__), Path(replay.__file__), Path(supported.__file__),
               OLD / "image_trace.py", replay.MOUNT, Path(entry.__file__), Path(runner.__file__),
               Path(replay.selections.__code__.co_filename), Path(replay.section_points.__code__.co_filename),
               Path(replay.project_reference_points.__code__.co_filename)}
    replay.write(OUT / "result.json", dict(seed=None, source_revision=replay.revision(),
                 confirmation_cut=replay.CONFIRM, replaced_from_m=replay.START,
                 source_sha256={str(p): replay.digest(p) for p in sorted(sources)},
                 tracks=results, termination="signed_contrast_and_fixed_control_replays_complete"))


if __name__ == "__main__":
    main()
