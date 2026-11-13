"""Score the saved Highlands recording with the pre-existing frozen observer."""
import argparse
from pathlib import Path
from unittest.mock import patch

from extract_frozen_corner_archive import run as extract
import extract_frozen_corner_archive as features
from fixed_left import old
from replay_independent import CACHE, load_fixed_policy, run, validate_layout_receipt, validate_transfer

out = Path(__file__).resolve().parent
capture = out / "highlands-capture/capture"
archives = capture / "archive-map.json"
proof = capture / "result.json"
motion = out / "highlands-motion"
track, layout = "ks_highlands", "layout_short"
receipt = load_fixed_policy()[0]
validate_transfer(track, layout)
validate_layout_receipt(archives, proof, track, layout)
reference = out.parent / "vision-vallelunga-local-corner-storage-review-20261112-01"
parity = reference / "eager-parity.json"
prior = old.read(parity)
assert old.digest(reference / "evaluate_eager.py") == prior["wrapper_sha256"]
assert max(row["max_abs_error"] for row in prior["modes"].values()) <= 1e-6
original_setup, original_pair = features.setup, features.extract_pair

def eager_setup(*args, **kwargs):
    network, projector, indices, pool, _, sources = original_setup(*args, **kwargs)
    return network, projector, indices, pool, network.simulate, sources

def progress(row, values, images):
    if row["capture_id"] % 100 == 0:
        print(f"FlyVis capture {row['capture_id']}", flush=True)

def report_pair(*args, **kwargs):
    return original_pair(*args, **kwargs, on_frame=progress)

with (patch.object(features, "setup", eager_setup), patch.object(features, "extract_pair", report_pair)):
    extract(archives, None, CACHE, motion, batch_pair=True, expected_track=track)
old.write(out / "highlands-motion-provenance.json", dict(
    integrator="original FlyVis Network.simulate, GPU eager",
    program_sha256=old.digest(Path(__file__)), frozen_sha256=old.digest(receipt),
    previous_two_frame_parity_sha256=old.digest(parity), capture_sha256=old.digest(proof),
    result_sha256={mode:old.digest(motion / mode / "flyvis-result.json")
                   for mode in ("original", "reflected")},
    seed=None, termination="archive_end"))
run(argparse.Namespace(archives=archives, capture_receipt=proof, track=track, layout=layout,
                       cache=CACHE, motion_extraction=motion, output=out / "highlands-replay"))
