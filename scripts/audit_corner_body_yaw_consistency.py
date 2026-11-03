"""Compare diagnostic heading change with native angular rates; never actor inputs."""

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
    parser.add_argument("--recording", required=True)
    parser.add_argument("--staged", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, staged = Path(args.recording), Path(args.staged)
    receipt, result = read(root / "result.json"), read(staged / "result.json")
    metadata, inputs = root / "diagnostic-frames.json", staged / "diagnostic-inputs.npz"
    if (receipt["status"] != "completed"
            or result["status"] != "completed_causal_fourview_body_diagnostic_stage"
            or sha256(metadata) != receipt["metadata_sha256"]
            or sha256(inputs) != result["diagnostic_inputs_sha256"]):
        raise ValueError("Original completed recording and staged diagnostic required")
    staged_sources = read(staged / "manifest.json")["config"]["source_hashes"]
    if (staged_sources.get(str(root / "result.json")) != sha256(root / "result.json")
            or staged_sources.get(str(metadata)) != sha256(metadata)):
        raise ValueError("Staged body inputs must originate from this exact recording")
    rows = read(metadata)
    times, forward, speed = [], [], []
    for i, row in enumerate(rows):
        source = Path(row["source_file"])
        if row["index"] != i or sha256(source) != row["sha256"]:
            raise ValueError("Original frame metadata identity and hash required")
        frame = read(source)
        diag = frame["airflow_diagnostic"]
        if diag != row["diagnostic"] or diag["actor_admissible"] is not False:
            raise ValueError("Exact isolated diagnostic metadata required")
        times.append(diag["started_source_clock"])
        forward.append(diag["body_forward_candidate"])
        speed.append(np.linalg.norm(diag["vehicle_velocity_candidate_m_s"]))
    times, forward, speed = map(np.asarray, (times, forward, speed))
    with np.load(inputs, allow_pickle=False) as z:
        body, valid = z["native_unregistered_body"].copy(), z["native_receipt_available"].copy()
        names = z["native_feature_names"].tolist()
    if (len(rows) != result["frames"] or body.shape != (len(rows), 21)
            or valid.shape != (len(rows),) or names[2:5] != ["angular_native_0",
                "angular_native_1", "angular_native_2"]
            or forward.shape != (len(rows), 3)
            or not all(np.isfinite(x).all() for x in (times, forward, speed, body))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Finite complete diagnostic body and ordered frame times required")
    # Azimuth is offline truth for comparison, never a model input.
    heading = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    dt = np.diff(times)
    rate = np.diff(heading) / dt
    axis = (body[:-1, 2:5] + body[1:, 2:5]) / 2
    eligible = valid[:-1] & valid[1:] & (speed[:-1] > 1) & (speed[1:] > 1) & (dt <= .15)
    x = rate[eligible]
    if len(x) < 3 or np.std(x) == 0:
        raise ValueError("Multiple moving observations with nonconstant rotation required")
    reports = []
    for i in range(3):
        y = axis[eligible, i]
        reports.append({"axis": i, "correlation": float(np.corrcoef(x, y)[0, 1]),
                        "slope_through_zero": float(np.dot(x, y) / np.dot(x, x)),
                        "rmse_vs_azimuth_rate": float(np.sqrt(np.mean((x-y)**2)))})
    subgroups = {}
    for label, mask in (("positive", eligible & (rate > .02)),
                        ("negative", eligible & (rate < -.02))):
        native = axis[:, 1][mask]
        target = rate[mask]
        subgroups[label] = {"count": int(mask.sum()),
                            "same_sign_count": int((target*native > 0).sum()),
                            "rmse": float(np.sqrt(np.mean((target-native)**2)))
                            if len(target) else None}
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "recording_result_sha256": sha256(root / "result.json"),
        "original_metadata_sha256": sha256(metadata), "staged_inputs_sha256": sha256(inputs),
        "diagnostic_only": True, "world_azimuth_is_not_actor_input": True})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = {"status": "completed_corner_yaw_consistency", "moving_intervals": len(x),
              "axes": reports, "axis_1_turn_subgroups": subgroups,
              "camera_physics_exposure_synchronization_verified": False,
              "native_source_sample_time_verified": False, "new_axis_admission": False,
              "actor_deployed": False, "optimizer_updates": 0}
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
