
"""Run locked scoring with the original eager FlyVis integrator after a GPU graph stall."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

import evaluate_recording as storage
import extract_frozen_corner_archive as extraction
from train_vision_teacher_line import digest, read, write

directory = Path(__file__).parent
protocol = storage.protocol
rehearsal = protocol / "gpu-feature-rehearsal"
original_setup = extraction.setup
original_pair = extraction.extract_pair
parity_receipt = directory / "eager-parity.json"


def progress(row, features, images):
    if row["capture_id"] % 100 == 0:
        print(f"FlyVis frame {row['capture_id']}", flush=True)


def report_pair(*args, **kwargs):
    return original_pair(*args, **kwargs, on_frame=progress)


def eager_setup(*args, **kwargs):
    network, projector, indices, pool, _, source = original_setup(*args, **kwargs)
    return network, projector, indices, pool, network.simulate, source


def verify_eager(cache, output):
    reference = read(protocol / "gpu-feature-rehearsal.json")
    assert reference["lock_sha256"] == digest(protocol / "lock.json")
    extraction.run(
        Path("runs/vision-silverstone1967-independent-capture-20261112-05/archive-map.json"),
        protocol / "gpu-rehearsal-index.jsonl", cache, output, batch_pair=True,
        expected_track=reference["track"],
    )
    results = {}
    for mode in ("original", "reflected"):
        expected = rehearsal / mode / "features.npz"
        actual = output / mode / "features.npz"
        assert digest(expected) == reference["modes"][mode]["features_sha256"]
        with np.load(expected, allow_pickle=False) as a, np.load(actual, allow_pickle=False) as b:
            assert np.array_equal(a["available"], b["available"]) and a["available"].all()
            delta = float(np.max(np.abs(a["features"] - b["features"])))
            assert delta <= 1e-6, (mode, delta)
        results[mode] = dict(max_abs_error=delta, eager_sha256=digest(actual),
                             frozen_graph_sha256=digest(expected))
    write(parity_receipt, dict(status="verified", track=reference["track"], frames=reference["frames"],
                              lock_sha256=reference["lock_sha256"], wrapper_sha256=digest(Path(__file__)),
                              modes=results))
    print(f"Original FlyVis backend parity verified: {results}", flush=True)


if __name__ == "__main__":
    with (patch.object(extraction, "setup", eager_setup),
          patch.object(extraction, "extract_pair", report_pair),
          patch.object(storage.admission, "verify_buffered_persistence_archive",
                       storage.verify_actual_writer)):
        cache = Path(sys.argv[sys.argv.index("--cache") + 1])
        verify_eager(cache, directory / "eager-silverstone-rehearsal")
        storage.entry.main()
        if len(storage.used) != 1:
            raise RuntimeError("Original MMF archive was not independently verified once")
