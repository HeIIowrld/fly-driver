"""Check archived paused-camera rotation diagnostics and four-view timing."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def audit(root, deferred):
    result = read(root / "result.json")
    if (
        result["status"] != "completed"
        or not result["all_hashes_verified"]
        or len(result["frames"]) != 26
        or not result["all_images_nonconstant"]
    ):
        raise ValueError("Completed original 26-frame capture required")
    for item in read(root / "archive.json"):
        if sha256(root / item["file"]) != item["sha256"]:
            raise ValueError("Archived camera file changed")
    spans, angles = [], []
    for index in range(26):
        frame = read(root / f"camera/frame-{index:04d}.json")
        diag = frame["camera_motion_diagnostic"]
        if (
            not frame["aid_probe_only"]
            or bool(frame["deferred_save_probe"]) != deferred
            or not diag["physical_body_paused"]
            or diag["actor_admissible"] is not False
            or diag["requested_yaw_left_rad_s"] != 0.2
            or len(frame["views"]) != 4
            or {v["view_id"] for v in frame["views"]} != {"front", "left", "right", "rear"}
        ):
            raise ValueError("Original paused rotation protocol and four views required")
        starts = [v["render_started_clock"] for v in frame["views"]]
        finishes = [v["render_call_finished_clock"] for v in frame["views"]]
        if deferred and any(v["save_finished_clock"] < max(finishes) for v in frame["views"]):
            raise ValueError("All renders must finish before save")
        spans.append((max(starts) - min(starts)) * 1000)
        angles.append(diag["yaw_left_angle_rad"])
    if min(angles) != 0 or max(angles) < 0.2 or np.any(np.diff(angles) < 0):
        raise ValueError("Original requested angular motion did not execute monotonically")
    differences = {}
    for view in ("front", "left", "right", "rear"):
        with Image.open(root / f"camera/0000-{view}.png") as img:
            first = np.asarray(img.convert("RGB"), dtype=np.int16)
        with Image.open(root / f"camera/0025-{view}.png") as img:
            last = np.asarray(img.convert("RGB"), dtype=np.int16)
        differences[view] = float(np.abs(last - first).mean())
    if any(value <= 0 for value in differences.values()):
        raise ValueError("Each original view must change during rotation")
    return {
        "groups": 26,
        "render_start_span_ms": spans,
        "median_ms": float(np.median(spans)),
        "p95_ms": float(np.percentile(spans, 95)),
        "max_ms": float(max(spans)),
        "requested_angle_range_rad": [min(angles), max(angles)],
        "nonzero_angle_frames": sum(a > 0 for a in angles),
        "first_to_last_rgb_mean_abs_change_0_255": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, candidate = Path(args.baseline), Path(args.candidate)
    paths = [
        baseline / "result.json",
        baseline / "archive.json",
        candidate / "result.json",
        candidate / "archive.json",
    ]
    pins = {str(p): sha256(p) for p in paths}
    base, deferred = audit(baseline, False), audit(candidate, True)
    if any(sha256(p) != h for p, h in pins.items()):
        raise ValueError("Original source changed")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "synthetic camera rotation; vehicle remains paused",
            "same_physical_exposure_or_vehicle_motion_not_verified": True,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_paused_rotation_deferred_save_comparison",
        "baseline": base,
        "deferred": deferred,
        "physical_vehicle_motion_tested": False,
        "actual_exposure_sync_verified": False,
        "optimizer_updates": 0,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
