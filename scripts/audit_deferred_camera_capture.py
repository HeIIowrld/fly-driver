"""Compare original paused probe groups with the bounded deferred-save diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def collect(root, deferred):
    result = read(root / "result.json")
    if (
        result["status"] != "completed"
        or result["all_hashes_verified"] is not True
        or result["all_images_nonconstant"] is not True
        or len(result["frames"]) != 4
    ):
        raise ValueError("Completed four-frame diagnostic required")
    archive = read(root / "archive.json")
    for item in archive:
        if sha256(root / item["file"]) != item["sha256"]:
            raise ValueError("Original archived image or metadata changed")
    spans = []
    for index in range(4):
        frame = read(root / f"camera/frame-{index:04d}.json")
        if (
            bool(frame.get("deferred_save_probe", False)) != deferred
            or len(frame["views"]) != 4
            or {x["view_id"] for x in frame["views"]} != {"front", "left", "right", "rear"}
        ):
            raise ValueError("Exact original view and condition required")
        starts = [v["render_started_clock"] for v in frame["views"]]
        finishes = [v["render_call_finished_clock"] for v in frame["views"]]
        if deferred and (
            not frame["aid_probe_only"]
            or any(v["save_finished_clock"] < max(finishes) for v in frame["views"])
        ):
            raise ValueError("Diagnostic and all-renders-before-save contract required")
        if deferred:
            for view in frame["views"]:
                with Image.open(root / "camera" / view["file"]) as image:
                    pixels = np.asarray(image.convert("RGB"))
                with Image.open(root / "camera" / view["reference_png"]) as image:
                    reference = np.asarray(image.convert("RGB"))
                if not np.array_equal(pixels, reference):
                    raise ValueError("Paired same-texture PNG pixels differ")
        spans.append((max(starts) - min(starts)) * 1000)
    return {
        "render_start_span_ms": spans,
        "median_ms": float(np.median(spans)),
        "maximum_ms": float(max(spans)),
        "observations": len(spans),
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
    left, right = collect(baseline, False), collect(candidate, True)
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original sources changed")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "four paired-format paused stationary groups per condition; unpaired in time",
            "not_physical_exposure_synchronization": True,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_paused_deferred_save_comparison",
        "baseline": left,
        "deferred_save_diagnostic": right,
        "all_paired_reference_pixels_exact": True,
        "moving_or_physical_exposure_validated": False,
        "optimizer_updates": 0,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
