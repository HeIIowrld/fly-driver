"""Audit diagnostic route metadata of TRAIN/development sources, never actor inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.data)
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            script_sha256=sha256(__file__),
            dataset_sha256=sha256(root / "dataset.json"),
            bins=100,
            interval_policy=(
                "Observed telemetry bins only; no interpolation across samples or episodes"
            ),
        ),
    )
    reports = []
    roles = {role: set() for role in ("train", "development")}
    for entry in read(root / "dataset.json")["episodes"]:
        assert entry["role"] in roles, "Evaluation data forbidden"
        assert sha256(root / entry["file"]) == entry["sha256"]
        receipt_file = Path(entry["original"]["recording"]) / "control/api-receipt.json"
        rows = [r for r in read(receipt_file)["telemetry"] if r["phase"] == "driving"]
        assert rows and all(r["ai_controlled"] and not r["paused"] for r in rows)
        route = [r["diagnostic_route"] for r in rows]
        assert all(r["actor_input"] is False for r in route)
        spline = np.array([r["spline_position"] for r in route])
        assert np.isfinite(spline).all() and np.all((spline >= 0) & (spline < 1))
        bins = set(np.floor(spline * 100).astype(int).tolist())
        roles[entry["role"]].update(bins)
        reports.append(
            dict(
                file=entry["file"],
                role=entry["role"],
                source_receipt_sha256=sha256(receipt_file),
                driving_telemetry_rows=len(rows),
                spline_start=float(spline[0]),
                spline_end=float(spline[-1]),
                start_end_lap_count=[route[0]["lap_count"], route[-1]["lap_count"]],
                observed_one_percent_bins=sorted(bins),
                apparent_wrap_transitions=int((np.diff(spline) < -0.5).sum()),
                lap_valid_true_samples=sum(r["lap_valid"] for r in route),
                wheels_outside_positive_samples=sum(r["wheels_outside"] > 0 for r in rows),
                maximum_telemetry_interval_seconds=float(
                    np.diff([r["script_clock"] for r in rows]).max()
                ),
                full_lap_verified=False,
            )
        )
    union = roles["train"] | roles["development"]
    write_json(
        run / "result.json",
        dict(
            status="completed_route_metadata_coverage_audit",
            reports=reports,
            observed_bins_by_role={k: sorted(v) for k, v in roles.items()},
            combined_observed_bins=sorted(union),
            missing_observed_bins=sorted(set(range(100)) - union),
            combined_observed_bin_count=len(union),
            bins_are_sample_coverage_not_continuous_distance=True,
            teacher_full_lap_verified=False,
            evaluation_role_arrays_or_receipts_opened=False,
            diagnostic_route_never_actor_input=True,
            optimizer_updates=0,
            ui_actions_sent=0,
            actuator_messages_sent=0,
            goal_achieved=False,
        ),
    )
    print(
        json.dumps(
            dict(combined_observed_bins=len(union), missing_bins=sorted(set(range(100)) - union))
        )
    )


if __name__ == "__main__":
    main()
