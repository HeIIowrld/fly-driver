"""Preserve a failed owned corner probe without treating it as successful training data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.recording)
    result, control = read(source / "result.json"), read(source / "control/api-receipt.json")
    if result["status"] != "failed" or control["status"] != "api_returned_false":
        raise ValueError("Original failed run and failed API receipt required")
    log_path = source / "live-frame-hashes.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf8").splitlines()]
    if not rows:
        raise ValueError("At least one original host-observed frame required")
    camera = Path.home() / "AppData/Local/flybrain/csp-camera" / result["request_id"]
    terminal = read(camera / "result.json")
    if terminal["request_id"] != result["request_id"] or terminal["frames"] != len(rows):
        raise ValueError("Exact stopped camera frame count required")
    paths = [source / "result.json", source / "control/api-receipt.json",
             source / "control/result.json", source / "control/native-motion-samples.json",
             source / "clock-initial.json", log_path, camera / "result.json"]
    pins = {str(p): sha256(p) for p in paths}
    run = begin_run(args.output, {"program_sha256": sha256(__file__), "source_hashes": pins,
                                  "scope": "failed run evidence only; no actor or training"})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    archive = run / "failed-diagnostic.zip"
    times, headings, speeds = [], [], []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        for path in paths:
            z.writestr("evidence/" + str(paths.index(path)) + "-" + path.name, path.read_bytes())
        for i, row in enumerate(rows):
            if row["index"] != i:
                raise ValueError("Exact original host frame sequence required")
            metadata = camera / f"frame-{i:04d}.json"
            data = metadata.read_bytes()
            if hashlib.sha256(data).hexdigest() != row["metadata_sha256"]:
                raise ValueError("Metadata changed since publication")
            frame = json.loads(data)
            if frame["request_id"] != result["request_id"] or frame["index"] != i:
                raise ValueError("Owned frame identity required")
            diag = frame["airflow_diagnostic"]
            if diag["actor_admissible"] is not False:
                raise ValueError("Diagnostic-only source required")
            f = diag["body_forward_candidate"]
            times.append(diag["started_source_clock"])
            headings.append(np.arctan2(f[0], f[2]))
            speeds.append(np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]))
            z.writestr("camera/" + metadata.name, data)
            if set(row["images"]) != {"front", "left", "right", "rear"}:
                raise ValueError("Four original view hashes required")
            for image in row["images"].values():
                path = camera / image["file"]
                if path.parent != camera or sha256(path) != image["sha256"]:
                    raise ValueError("Image differs from original host-observed hash")
                z.write(path, "camera/" + path.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Stored archive CRC check failed")
        for row in rows:
            for image in row["images"].values():
                if hashlib.sha256(z.read("camera/" + image["file"])).hexdigest() != image["sha256"]:
                    raise ValueError("Archived image hash differs")
    times, headings, speeds = map(np.asarray, (times, headings, speeds))
    if (not all(np.isfinite(x).all() for x in (times, headings, speeds))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Finite ordered original frames required")
    heading = np.unwrap(headings)
    dt = np.diff(times)
    rate = np.diff(heading) / dt
    eligible = (speeds[:-1] > 1) & (speeds[1:] > 1) & (dt <= 0.2)
    report = {"status": "preserved_failed_diagnostic", "frames": len(rows),
              "stop_reason": control["stop_reason"], "moving_intervals": int(eligible.sum()),
              "heading_span_rad": float(np.ptp(heading)),
              "positive_rate_over_0_02": int((rate[eligible] > .02).sum()),
              "negative_rate_below_minus_0_02": int((rate[eligible] < -.02).sum()),
              "archive_sha256": sha256(archive), "archive_images_verified": True,
              "source_retained": True, "training_admissible": False,
              "physical_sync_verified": False, "actor_deployed": False}
    write_json(run / "result.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
