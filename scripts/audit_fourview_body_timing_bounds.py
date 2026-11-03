"""Audit saved camera render-start bounds against original native body receipts."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, audit = Path(args.recording), Path(args.audit)
    result, proof, manifest = (
        read(root / "result.json"),
        read(audit / "result.json"),
        read(audit / "manifest.json"),
    )
    events_path, archive_path = root / "events.jsonl", root / "vision/camera.zip"
    bundle = read(root / "vision/bundle.json")
    if (
        result["status"] != "completed"
        or proof["status"] != "completed"
        or not proof["producer_time_metadata_replay_verified"]
        or manifest["config"]["events_sha256"] != sha256(events_path)
        or bundle["sha256"] != sha256(archive_path)
    ):
        raise ValueError("Verified original recording and audit required")
    inputs = [
        events_path,
        archive_path,
        root / "vision/bundle.json",
        audit / "result.json",
        audit / "manifest.json",
    ]
    pins = {str(p): sha256(p) for p in inputs}
    events = [json.loads(line) for line in events_path.read_text(encoding="utf8").splitlines()]
    values, cross_view_spans, body_ages = [], [], []
    with zipfile.ZipFile(archive_path) as archive:
        initial = json.loads(archive.read("clock-initial.json"))
        clock = CSPClockBounds(
            initial["generation"],
            maximum_width=initial["maximum_width_seconds"],
            host_quantization=initial.get("host_clock_quantization_seconds", 0),
        )
        for ping in initial["exchanges"]:
            clock.observe(
                ping["sent"], ping["received"], ping["producer_clock"], initial["generation"]
            )
        if clock.record() != initial:
            raise ValueError("Original clock bounds do not replay exactly")
        for event in events:
            camera, native = event["camera"], event["native"]
            if camera["generation"] != clock.generation:
                raise ValueError("Original generation changed")
            raw = archive.read(Path(camera["file"]).name)
            if hashlib.sha256(raw).hexdigest() != camera["file_sha256"]:
                raise ValueError("Original camera metadata changed")
            metadata = json.loads(raw)
            by_view = {v["view_id"]: v for v in metadata["views"]}
            if set(by_view) != {"front", "left", "right", "rear"}:
                raise ValueError("Four original camera views required")
            body_receipt = native["received_monotonic_seconds"]
            decision = event["decision_monotonic_seconds"]
            if body_receipt > decision or not 0 <= decision - body_receipt <= 0.05:
                raise ValueError("Original body receipt must be causal and fresh")
            body_ages.append(decision - body_receipt)
            source_times = []
            for view in camera["views"]:
                name = view["view_id"]
                source = by_view[name]["render_started_clock"]
                if source != view["producer_render_started"]:
                    raise ValueError("Original producer render time changed")
                lower = clock.render_start_lower(source, clock.generation)
                upper = source + clock.high + clock.host_quantization
                if abs(lower - view["render_started_lower"]) > 1e-8:
                    raise ValueError("Recorded host lower bound differs")
                values.append([name, body_receipt - upper, body_receipt - lower])
                source_times.append(source)
            cross_view_spans.append(max(source_times) - min(source_times))
    if len(events) != result["decisions"] or len(values) != len(events) * 4:
        raise ValueError("All original observations and views required")
    reports = []
    for name in ("front", "left", "right", "rear"):
        v = np.array([[r[1], r[2]] for r in values if r[0] == name])
        reports.append(
            {
                "view": name,
                "observations": len(v),
                "body_receipt_minus_render_start_lower_bound_ms_median": float(
                    np.median(v[:, 0]) * 1000
                ),
                "body_receipt_minus_render_start_upper_bound_ms_median": float(
                    np.median(v[:, 1]) * 1000
                ),
                "body_receipt_after_render_start_proven_fraction": float((v[:, 0] > 0).mean()),
                "upper_bound_ms_max": float(v[:, 1].max() * 1000),
            }
        )
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original evidence changed")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "host receipt and producer render-start bounds, not physical exposure",
            "clock": "initial conservative offset range; source unit-rate assumption",
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_fourview_body_timing_bounds_audit",
        "reports": reports,
        "observations": len(events),
        "clock_initial_width_ms": clock.high * 1000 - clock.low * 1000,
        "cross_view_render_start_span_median_ms": float(np.median(cross_view_spans) * 1000),
        "cross_view_render_start_span_max_ms": float(max(cross_view_spans) * 1000),
        "body_receipt_age_median_ms": float(np.median(body_ages) * 1000),
        "physical_camera_exposure_or_native_sample_time_verified": False,
        "optimizer_updates": 0,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
