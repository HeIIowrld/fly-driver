"""Extract verified relative travel for comparison of separate boundary onset estimates."""
from pathlib import Path
import argparse

import numpy as np

from evaluate_fused_corner import frame_key, load_score
from evaluate_fused_corner_warning_end import capture_motion
from profile_replay import TRACKS, archive_path, sources
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent
BOUNDARY = Path("runs/vision-corner-boundary-hinge-20261111-01")


def run(track):
    final, base, _ = sources(track)
    final_path, base_path = final / "result.json", base / "result.json"
    proof = read(final_path)
    archive = archive_path(track)
    if digest(archive) != read(base_path)["archives_sha256"]:
        raise ValueError(f"Capture map changed: {track}")
    rows = load_score(final / "score", proof["score_receipt_sha256"])
    values = list(capture_motion(archive, track, proof["camera_bundle_sha256_by_run"]))
    if [key for key, _ in values] != [frame_key(row) for row in rows]:
        raise ValueError(f"Motion frame order differs: {track}")
    path = OUT / f"{track}-motion.npz"
    np.savez_compressed(path, motion=np.array([m if m is not None else np.nan for _, m in values]),
                        run=np.array([r["run"] for r in rows]), frame=np.array([r["frame"] for r in rows]))
    inputs = (Path(__file__), Path("scripts/evaluate_fused_corner_warning_end.py"),
              archive, final_path, base_path, BOUNDARY / f"{track}-sections.json")
    write(path.with_suffix(".json"), dict(track=track, revision=proof["revision"], frames=len(rows),
          motion_sha256=digest(path), input_sha256={str(p):digest(p) for p in inputs},
          termination="verified_relative_travel"))
    print(track, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="+", choices=TRACKS)
    for name in parser.parse_args().tracks:
        run(name)
