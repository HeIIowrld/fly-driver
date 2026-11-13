"""Verify the fresh Imola archive and label its admission without inference."""
from collections import Counter
from pathlib import Path

from check_corner_stability_coverage import coverage
from evaluate_live_corner_geometry import common_rows, load_track_truth
from evaluate_fused_corner import load_frozen
from evaluate_independent_corner import admission
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.frame_persistence_profile_v3 import verify_buffered_persistence_archive
from train_vision_teacher_line import digest, jsonl, read, write

settings = Path(__file__).parent
run = Path("runs/vision-imola-approach-capture-20261111-02")
receipt_path = run/"result.json"
receipt = read(receipt_path)
assert receipt["status"] == "captured"
assert receipt["closed"]["status"] == "stopped"
assert not receipt["closed"]["error"] and not receipt["closed"]["worker_alive"]
assert receipt["closed"]["producer_terminal"]["status"] == "completed"
assert receipt["closed"]["producer_terminal"]["frames"] == receipt["frames"]
assert receipt["archive"]["all_hashes_verified"]
assert receipt["actor_control_sent"] is False
capture = settings/"capture-buffered.json"
assert receipt["protocol_sha256"] == digest(capture)
assert receipt["frames"] == load_frozen(capture)["capture"]["max_frames"]
track_path = settings/"track.json"
track = load_frozen(track_path)
geometry = Path(next(iter(track["hashes"])))
truth, _, _ = load_track_truth(geometry)
mapping = run/"archive-map.json"
with CameraArchive(read(mapping)["imola"]) as archive:
    assert archive.bundle_sha256 == receipt["archive"]["bundle_sha256"]
    stored = verify_buffered_persistence_archive(archive,expected_mmf=True)
    rows = list(common_rows("imola",archive,{}, "imola",truth))
assert len(stored)==len(rows)==receipt["frames"]
base = settings/"diagnostic-base.json"
requirements = load_frozen(base)["data_admission"]
minimum = requirements["minimum_distinct_physical_corners_per_side"]
samples = dict(Counter(r["label"] for r in rows if r["truth"]=="approach" and r["camera_complete"]))
out = run/"admission"
out.mkdir()
jsonl(out/"rows.jsonl", rows)
result = dict(status="checked", approach_frames_by_side=samples,
              admission=admission(rows,requirements), stable_coverage=coverage(rows,minimum),
              receipt_sha256=digest(receipt_path), archives_sha256=digest(mapping),
              geometry_sha256=digest(geometry), rows_sha256=digest(out/"rows.jsonl"),
              program_sha256=digest(Path(__file__)), termination="archive_and_admission_verified")
write(out/"result.json",result)
print(result)
