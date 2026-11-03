"""Synthetic RGB recording/replay fixture. Does not enumerate or acquire any real window."""

import argparse
import hashlib
import json

import numpy as np

from flybrain_drive.ac_capture import WindowCapture, WindowState
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from scripts.record_ac_capture import record_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = begin_run(args.output, {"scope": "synthetic RGB acquisition fixture, no real window",
                                 "seed": 760101, "game_runtime_verified": False})
    state = WindowState(17, 42, "/synthetic/acs.exe", 0, 0, 64, 32)
    counter = 0

    def pixels(_):
        nonlocal counter
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        image[2:20, 2 + counter:4 + counter] = [220, 10, 50]
        counter += 1
        return image

    capture = WindowCapture(lambda: state, pixels)
    result = record_frames(capture, run, frames=4, rate=25, max_bytes=1024**2)
    rows = [json.loads(line) for line in (run / "captures.jsonl").read_text().splitlines()]
    for index, row in enumerate(rows):
        path = run / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["file_sha256"]
        image = load_array(path, allow_pickle=False)
        assert hashlib.sha256(image.tobytes()).hexdigest() == row["rgb_sha256"]
        assert row["capture_id"] == index and row["acquisition_kind"] == "synthetic_fixture"
        assert row["engine_frame_id"] is None and row["simulation_seconds"] is None
        assert not image[-1].any()
    result.update(status="completed", fixture_only=True, rgb_replay_hashes_verified=len(rows))
    write_json(run / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
