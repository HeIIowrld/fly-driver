"""Compare same-history common versus per-view body contexts using a fixed probe."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("common", "per-view", "per-view-audit", "output"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    common, specific, source = map(Path, (a.common, a.per_view, a.per_view_audit))
    results = [read(path / "result.json") for path in (common, specific)]
    manifests = [read(path / "manifest.json")["config"] for path in (common, specific)]
    source_result = read(source / "result.json")
    for path, result in zip((common, specific), results, strict=True):
        if (
            result["status"] != "completed_pace_visual_body_counterfactual"
            or result["fixed_yaw_probe"] is not True
            or result["training_admissible"] is not False
            or sha256(path / "events.json") != result["events_sha256"]
        ):
            raise ValueError("Unchanged fixed-probe diagnostic replay required")
    keys = (
        "events_sha256",
        "vision_sha256",
        "body_sha256",
        "context_sha256",
        "brake_sha256",
        "flight_sha256",
        "opponent_sha256",
        "seed",
        "fixed_task_speed_kmh",
        "fixed_yaw_probe_weight",
        "program_sha256",
    )
    if (
        any(manifests[0][key] != manifests[1][key] for key in keys)
        or manifests[0]["per_view_context_sha256"] is not None
        or manifests[1]["per_view_context_sha256"] != source_result["arrays_sha256"]
        or sha256(source / "per-view-body.npz") != source_result["arrays_sha256"]
        or source_result["status"] != "completed_pace_per_view_body_receipt_audit"
    ):
        raise ValueError("Only original historical body selection may differ")
    histories = [read(path / "events.json") for path in (common, specific)]
    if len(histories[0]) != 731 or len(histories[1]) != 731:
        raise ValueError("Complete original 731 decisions required")
    for x, y in zip(*histories, strict=True):
        if any(x[key] != y[key] for key in ("index", "time", "packet", "moving_brake")):
            raise ValueError("Original decision sequence differs")
    delta = np.abs(
        np.asarray([x["intents"][0] for x in histories[0]])
        - np.asarray([x["intents"][0] for x in histories[1]])
    )
    if delta.shape != (731, 2) or not np.isfinite(delta).all():
        raise ValueError("Finite paired outputs required")
    brakes = np.asarray([x["moving_brake"] for x in histories[0]], dtype=bool)
    if brakes.sum() != 62:
        raise ValueError("Original independently audited moving-brake count changed")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "common_result_sha256": sha256(common / "result.json"),
            "per_view_result_sha256": sha256(specific / "result.json"),
            "per_view_audit_sha256": sha256(source / "result.json"),
            "seed": None,
        },
    )
    summary = {
        "status": "completed_pace_per_view_probe_audit",
        "decisions": len(delta),
        "distinct_source_packet_groups": source_result[
            "decision_groups_with_distinct_view_packets"
        ],
        "output_nonzero_groups": int(np.any(delta != 0, axis=1).sum()),
        "mean_abs": delta.mean(axis=0).tolist(),
        "p95_abs": np.percentile(delta, 95, axis=0).tolist(),
        "max_abs": delta.max(axis=0).tolist(),
        "braking_max_abs": delta[brakes].max(axis=0).tolist(),
        "numerical_comparison_tolerance": 2e-6,
        "difference_exceeds_numerical_tolerance": bool(np.max(delta) > 2e-6),
        "probe_is_hand_chosen_not_measured_or_trained": True,
        "physical_source_or_exposure_time_verified": False,
        "fit_or_control": False,
    }
    write_json(run / "result.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
