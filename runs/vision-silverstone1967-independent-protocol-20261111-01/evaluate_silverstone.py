"""Evaluate the predeclared Silverstone 1967 comparisons on an admitted archive."""
import argparse
from pathlib import Path

import numpy as np

from anchor_distance import evaluate, filter_distances
from audit_corner_capture_admission import review
from evaluate_fused_corner import frame_key, load_frozen, load_score
from evaluate_fused_corner_warning_end import capture_motion
from evaluate_recovered_corner import run as run_visual
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
TRACK = "ks_silverstone1967"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    for name in ("capture-dir", "track-dir", "cache", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    lock = load_frozen(OUT / "lock.json")
    base = load_frozen(OUT / "base.json")
    candidate = load_frozen(OUT / "candidate.json")
    acquisition = load_frozen(OUT / "capture.json")
    rules = read(OUT / "admission-review.json")
    if (base["proposed_track"] != TRACK or TRACK in base["excluded_development_tracks"]
            or rules["immutable_candidate_sha256"] != digest(OUT/"candidate.json")
            or rules["capture_protocol_sha256"] != digest(OUT/"capture.json")
            or candidate["development_rehearsal"]
            or acquisition["capture"]["period_s"] != .09):
        raise ValueError("Silverstone protocol scope differs from frozen acquisition")
    if args.verify_only:
        print("Frozen Silverstone protocol hashes, track, gates and 90ms acquisition verified")
        return
    if any(p is None for p in (args.capture_dir, args.track_dir, args.cache, args.output)):
        parser.error("Capture, track, cache and output paths are required")
    if args.track_dir.name != TRACK:
        raise ValueError("Frozen evaluation requires Silverstone 1967")
    expected = [Path(p) for p in base["hashes"] if Path(p).name == "fast_lane.ai"]
    if expected != [(args.track_dir / "ai/fast_lane.ai").resolve()]:
        raise ValueError("Track installation differs from pinned geometry")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    admitted = review(args.capture_dir, OUT/"capture.json", OUT/"admission-review.json", output/"admission")
    if not admitted["ready_for_scoring"]:
        write(output/"result.json", dict(status="not_admitted", admission=admitted,
              termination="independent_admission"))
        return
    raw = output/"visual"
    archive = args.capture_dir/"archive-map.json"
    run_visual(argparse.Namespace(protocol=OUT/"candidate.json", archives=archive,
                   track_dir=args.track_dir, cache=args.cache, output=raw, base_replay=None))
    receipt, replay = read(raw/"result.json"), read(raw/"base/result.json")
    if (receipt["status"] != "evaluated" or receipt["development_rehearsal"]
            or replay["admission_rows_sha256"] != admitted["rows_sha256"]):
        raise ValueError("Visual timeline differs from independently admitted frames")
    original = load_score(raw/"base/parent/score", replay["score_receipt_sha256"]["parent"])
    visual = load_score(raw/"score", receipt["score_receipt_sha256"])
    timing = list(capture_motion(archive, TRACK, receipt["camera_bundle_sha256_by_run"]))
    if [key for key, _ in timing] != [frame_key(r) for r in visual]:
        raise ValueError("Body-relative motion differs from the admitted timeline")
    settings = tuple(lock["exploratory_distance_filter"])
    motion = np.array([step if step is not None else np.nan for _,step in timing])
    filtered = list(filter_distances(visual, motion, settings))
    group = dict(original=original, rules=base["comparison"])
    visual_report = evaluate(group, visual)
    if visual_report["metrics"] != receipt["comparison"]["metrics"]:
        raise ValueError("Frozen visual metrics changed in distance evaluation")
    report = evaluate(group, filtered)
    for field in ("left", "right", "outside", "missing"):
        if report["metrics"]["candidate"][field] != visual_report["metrics"]["candidate"][field]:
            raise ValueError("Distance comparison changed a direction or warning state")
    jsonl(output/"filtered-rows.jsonl", filtered)
    write(output/"result.json", dict(status="evaluated", track=TRACK, admission_sha256=digest(output/"admission/result.json"),
          visual=visual_report, mixed=replay["comparison"], distance_filter=report,
          comparison_protocol_sha256=digest(OUT/"lock.json"),
          filtered_rows_sha256=digest(output/"filtered-rows.jsonl"),
          termination="silverstone_independent_comparisons_complete"))


if __name__ == "__main__":
    main()
