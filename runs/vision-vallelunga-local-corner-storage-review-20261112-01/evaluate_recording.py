"""Apply the existing MMF verifier for the actual eight-slot writer."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import audit_corner_capture_admission as admission
from flybrain_drive.immutable_frame_writer import verify_persistence_archive
from train_vision_teacher_line import digest, write

protocol = Path("runs/vision-vallelunga-local-corner-protocol-20261112-01")
spec = importlib.util.spec_from_file_location("frozen_vallelunga_evaluation", protocol/"evaluate.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
used = []

def verify_actual_writer(camera, *, expected_mmf):
    journal = camera.json("frame-persistence-journal.json")
    if expected_mmf is not True or journal.get("bounded_profile") is not None or journal.get("maximum_pending") != 8:
        raise ValueError("Expected original eight-slot MMF persistence journal")
    rows = verify_persistence_archive(camera, expected_mmf=True)
    used.append(dict(groups=len(rows), maximum_pending=8, maximum_queued=journal["maximum_queued"],
                     bundle_sha256=camera.bundle_sha256))
    write(Path(__file__).parent/"storage-verification.json", dict(status="verified_original_mmf_writer",
        storage=used[0], evaluator_sha256=digest(protocol/"evaluate.py"),
        original_verifier_sha256=digest(Path(verify_persistence_archive.__code__.co_filename)),
        recovery_sha256=digest(Path(__file__))))
    return rows

if __name__ == "__main__":
    with patch.object(admission, "verify_buffered_persistence_archive", verify_actual_writer):
        entry.main()
    if len(used) != 1:
        raise RuntimeError("Original MMF archive was not independently verified once")
