"""Replay image tracing with directly supported seeds and slanted contrast checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import replay as old
from image_trace import FrontTrace, MIN_CONTRAST, trace


def directional_evidence(image, side):
    lab = old.cv2.cvtColor(image, old.cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = old.cv2.blur(lab, (3, 1))
    scores = []
    for dx, dy in ((3 * side, 0), (2 * side, -2), (side, -3)):
        inside = np.roll(lab, (dy, dx), axis=(0, 1))
        outside = np.roll(lab, (-dy, -dx), axis=(0, 1))
        delta = outside - inside
        inner_chroma = np.linalg.norm(inside[:, :, 1:] - 128., axis=2)
        outer_chroma = np.linalg.norm(outside[:, :, 1:] - 128., axis=2)
        chroma = np.linalg.norm(delta[:, :, 1:], axis=2)
        scores.append(chroma + .35 * np.abs(delta[:, :, 0])
                      + .25 * np.clip(outer_chroma - inner_chroma, -8., 16.))
    evidence = np.maximum.reduce(scores)
    evidence[:5] = 0
    evidence[-5:] = 0
    evidence[:, :5] = 0
    evidence[:, -5:] = 0
    return evidence


class SupportedTrace(FrontTrace):
    def curves(self, image, boundaries):
        if image.shape[:2] != (self.height, self.width):
            raise ValueError("Unexpected front camera shape")
        result = np.full((self.height, 2), np.nan)
        last = int(np.floor(self.rows[-1])) - 3
        far_start = int(np.ceil(self.rows[7]))
        for index, side in enumerate((1, -1)):
            x = .5 * (self.width - 1) - self.scale * (boundaries[:, index] - self.origin[1])
            eligible = np.flatnonzero(np.isfinite(x[:11]) & (x[:11] >= 9.) & (x[:11] <= self.width - 10.)
                                      & (self.rows[:11] < self.height - 5))
            if not len(eligible):
                continue
            evidence = directional_evidence(image, side)
            for i in eligible:
                row, column = int(round(self.rows[i])), int(round(x[i]))
                if np.max(evidence[row, column - 5:column + 6]) < MIN_CONTRAST:
                    continue
                curve = trace(evidence, (x[i], self.rows[i]), last)
                if np.isfinite(curve[last:far_start + 1]).sum() >= 4:
                    result[:, index] = curve
                    break
        return result


def main():
    old.cv2.setNumThreads(1)
    original = old.OUT
    old.OUT = original / "supported"
    old.OUT.mkdir(exist_ok=True)
    old.FrontTrace = SupportedTrace
    entry = old.context()
    replay = entry.import_replay()
    mount = old.read(old.MOUNT)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {t: executor.submit(old.replay_track, t, entry, replay, mount) for t in replay.CHECK}
        results = {}
        for track, job in jobs.items():
            results[track] = job.result()
            print(track, results[track]["metrics"], flush=True)
    sources = {Path(__file__), Path(old.__file__), old.MOUNT, original / "image_trace.py",
               Path(entry.__file__), Path(replay.__file__), Path(old.selections.__code__.co_filename),
               Path(old.section_points.__code__.co_filename), Path(old.project_reference_points.__code__.co_filename)}
    old.write(old.OUT / "result.json", dict(seed=None, source_revision=old.revision(),
              confirmation_cut=old.CONFIRM, replaced_from_m=old.START, tracks=results,
              source_sha256={str(p): old.digest(p) for p in sorted(sources)},
              termination="all_saved_frames_scored_with_control_parity"))
    import verify
    verify.OUT = old.OUT
    verify.RUNS = original.parent
    verify.main()


if __name__ == "__main__":
    main()
