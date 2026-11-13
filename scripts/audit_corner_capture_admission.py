"""Verify a buffered four-view capture and count scoreable corner approaches."""
import argparse
from collections import Counter
from pathlib import Path

from check_corner_stability_coverage import coverage
from evaluate_fused_corner import load_frozen
from evaluate_independent_corner import admission
from evaluate_live_corner_geometry import common_rows, load_track_truth
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.frame_persistence_profile_v3 import verify_buffered_persistence_archive
from train_vision_teacher_line import digest, jsonl, read, write


def review(capture_dir, protocol_path, review_path, output):
    frozen = read(review_path)
    if (frozen["capture_protocol_sha256"] != digest(protocol_path)
            or frozen["checker_sha256"] != digest(Path("scripts/check_corner_stability_coverage.py"))):
        raise ValueError("Capture or coverage checker differs from the frozen review")
    acquisition = load_frozen(protocol_path)
    route = load_frozen(Path(acquisition["warning_end_protocol"]))
    acquired = load_frozen(Path(route["fusion_protocol"]))
    acquisition_base = load_frozen(Path(acquired["base_protocol"]))
    candidate_path = Path(frozen.get("evaluation_protocol", route["fusion_protocol"]))
    if frozen["immutable_candidate_sha256"] != digest(candidate_path):
        raise ValueError("Candidate differs from the frozen review")
    candidate = load_frozen(candidate_path)
    base = load_frozen(Path(candidate["base_protocol"]))
    track = base["proposed_track"]
    if track != acquisition_base["proposed_track"] or candidate["camera_config"] != acquired["camera_config"]:
        raise ValueError("Capture and evaluation must use the same track and camera")
    geometry = [Path(path) for path in base["hashes"] if Path(path).name == "fast_lane.ai"]
    if len(geometry) != 1:
        raise ValueError("Expected one pinned track geometry file")
    truth, _, _ = load_track_truth(geometry[0])

    receipt_path = capture_dir/"result.json"
    receipt = read(receipt_path)
    closed = receipt["closed"]
    terminal = closed["producer_terminal"]
    expected = acquisition["capture"]["max_frames"]
    if (receipt["status"] != "captured" or receipt["track"] != track
            or receipt["protocol_sha256"] != digest(protocol_path)
            or receipt["actor_control_sent"] is not False or not all(receipt["readiness"].values())
            or closed["status"] != "stopped" or closed["error"] or closed["worker_alive"]
            or terminal["status"] != "completed" or terminal["frames"] != expected
            or receipt["frames"] != expected or not receipt["archive"]["all_hashes_verified"]):
        raise ValueError("Capture did not close with every expected immutable group")
    mapping = capture_dir/"archive-map.json"
    entries = read(mapping)
    if set(entries) != {track}:
        raise ValueError("Expected one original capture for the frozen track")
    with CameraArchive(entries[track]) as archive:
        if archive.bundle_sha256 != receipt["archive"]["bundle_sha256"]:
            raise ValueError("Camera bundle differs from its terminal receipt")
        stored = verify_buffered_persistence_archive(archive, expected_mmf=True)
        rows = list(common_rows(track, archive, {}, track, truth))
    if len(stored) != expected or len(rows) != expected:
        raise ValueError("Persisted and scoreable timelines disagree with the capture count")

    stable = coverage(rows, frozen["minimum_consecutive_corners_per_side"])
    check = admission(rows, base["data_admission"])
    counts = Counter(row["label"] for row in rows if row["truth"] == "approach" and row["camera_complete"])
    output.mkdir(parents=True, exist_ok=False)
    jsonl(output/"rows.jsonl", rows)
    result = dict(track=track, status="checked", approach_frames_by_side=counts,
                  admission=check, stable_coverage=stable,
                  ready_for_scoring=check["passed"] and stable["minimum_coverage_met"],
                  receipt_sha256=digest(receipt_path), archives_sha256=digest(mapping),
                  review_sha256=digest(review_path), rows_sha256=digest(output/"rows.jsonl"),
                  program_sha256=digest(Path(__file__)), termination="archive_and_admission_verified")
    write(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture_dir", "protocol_path", "review_path", "output"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    result = review(**vars(parser.parse_args()))
    print(result)


if __name__ == "__main__":
    main()
