"""Audit preserved four-camera render-start bounds without claiming exposure sync."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    bundle = read(root / "vision/bundle.json")
    archive = root / "vision" / bundle["file"]
    manifest = root / "vision/archive.json"
    if (sha256(archive) != bundle["sha256"]
            or sha256(manifest) != bundle["archive_manifest_sha256"]):
        raise ValueError("Original archived camera evidence hash mismatch")
    events = root / "events.jsonl"
    pins = {str(p): sha256(p) for p in (events, archive, manifest, root / "vision/bundle.json",
                                       root / "result.json")}
    with zipfile.ZipFile(archive) as z:
        initial = json.loads(z.read("clock-initial.json"))
        final = json.loads(z.read("clock-final.json"))
    if initial["generation"] != final["generation"]:
        raise ValueError("Clock generation mismatch")
    rows = [json.loads(line) for line in events.open(encoding="utf8")]
    cameras = [(r["decision_monotonic_seconds"], r["camera"]) for r in rows
               if "camera" in r and "decision_monotonic_seconds" in r]
    if not cameras:
        raise ValueError("No original decision camera records")
    seen, gaps, skews, ages, repeated = {}, [], [], [], 0
    last_time, last_id = None, None
    for decision, camera in cameras:
        if camera["generation"] != initial["generation"]:
            raise ValueError("Camera generation changed")
        capture = camera["capture_id"]
        views = camera["views"]
        if [v["view_id"] for v in views] != ["front", "left", "right", "rear"]:
            raise ValueError("Camera view order changed")
        identity = tuple((v["file_sha256"], v["rgb_sha256"], v["render_started_lower"])
                         for v in views)
        if capture in seen and seen[capture] != identity:
            raise ValueError("Repeated capture identity mismatch")
        seen[capture] = identity
        starts = np.array([v["render_started_lower"] for v in views], dtype=float)
        if not np.isfinite(starts).all() or not np.isfinite(decision):
            raise ValueError("Finite original clocks required")
        if last_time is not None:
            if decision <= last_time or capture < last_id:
                raise ValueError("Decision or capture time moved backwards")
            gaps.append(decision-last_time)
            repeated += int(capture == last_id)
        ages.extend((decision-starts).tolist())
        if (decision-starts < 0).any():
            raise ValueError("Render lower bound after decision")
        skews.append(float(starts.max()-starts.min()))
        last_time, last_id = decision, capture
    if len(cameras) < 2:
        raise ValueError("Two or more decisions required")
    run = begin_run(args.output, {"program_sha256": sha256(__file__), "inputs": pins,
        "actor_deployed": False, "optimizer_updates": 0})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = {"status": "completed_four_camera_render_bound_audit",
        "decisions": len(cameras), "unique_capture_ids": len(seen),
        "consecutive_reused_captures": repeated, "decision_gap_seconds": stats(gaps),
        "per_decision_render_lower_spread_seconds": stats(skews),
        "per_view_render_start_age_upper_seconds": stats(ages),
        "clock_initial_offset_lower": initial["offset_lower_seconds"],
        "clock_final_offset_lower": final["offset_lower_seconds"],
        "clock_initial_uncertainty_seconds": initial["uncertainty_seconds"],
        "clock_final_uncertainty_seconds": final["uncertainty_seconds"],
        "render_bound_spread_is_not_physical_exposure_skew": True,
        "recorded_camera_body_ray_registration_verified": False,
        "physical_exposure_sync_verified": False, "optimizer_updates": 0,
        "actor_deployed": False, "goal_achieved": False}
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original evidence changed")
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
