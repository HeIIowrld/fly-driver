"""Reconstruct fixed motion readout and source clocks for actual/counterfactual probes."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--original-replay", required=True)
    parser.add_argument("--held-replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    collection, original, held = map(
        Path, (args.collection, args.original_replay, args.held_replay)
    )
    cases = read(collection / "result.json")["conditions"]
    replay_reports = [read(p / "result.json") for p in (original, held)]
    for report in replay_reports:
        assert report["status"] == "completed_fixed_author_camera_motion_replay"
        assert not report["author_weights_changed"]
    original_arguments = read(original / "manifest.json")["config"]["arguments"]
    assert not original_arguments.get("hold_first_retina", False)
    if "neural_input_intervention" in replay_reports[0]:
        assert replay_reports[0]["neural_input_intervention"] == "original_video"
    assert replay_reports[1]["neural_input_intervention"] == "hold_first_retina_counterfactual"
    config = read(original / "manifest.json")["config"]
    model = Path(config["arguments"]["model_dir"])
    calibration = Path(config["arguments"]["calibration"])
    assert sha256(model / "best_chkpt") == config["author_checkpoint_sha256"]
    assert sha256(calibration / "calibration.npz") == config["fixed_synthetic_calibration_sha256"]
    with load_array(calibration / "calibration.npz", allow_pickle=False) as archive:
        center, scale, coefficients = (archive[k] for k in ("center", "scale", "coefficients"))
    # Recover exactly the original projector rays from its frozen configuration/UV lattice.
    replay_registration = Path(
        "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1346-v2-visual-body-replays"
        "/episode-00-train/vision/retina-registration.npz"
    )
    with load_array(replay_registration, allow_pickle=False) as archive:
        pool = VisualSelfMotionPool(
            read("configs/ac-csp-yaw-retina.json"), archive["body_directions"]
        )
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            arguments=vars(args),
            script_sha256=sha256(__file__),
            original_replay_sha256=sha256(original / "result.json"),
            held_replay_sha256=sha256(held / "result.json"),
            optimizer_updates=0,
            retrospective_known_scene_diagnosis=True,
            actor_admissible=False,
        ),
    )
    results, all_handles = [], []
    for index, case in enumerate(cases):
        source = Path(case["output"])
        rows = read(source / "rows.json")
        handles = []
        white = []
        with zipfile.ZipFile(source / "vision/camera.zip") as archive:
            for entry in read(source / "vision/archive.json"):
                raw = archive.read(entry["file"])
                assert len(raw) == entry["bytes"]
                assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
            source_clocks = []
            for row in rows:
                raw = archive.read(row["file"])
                assert hashlib.sha256(raw).hexdigest() == row["file_sha256"]
                frame = json.loads(raw)
                assert (
                    frame["aid_probe_only"]
                    and frame["camera_motion_diagnostic"]["actor_admissible"] is False
                )
                source_clocks.append(frame["views"][0]["render_started_clock"])
                if row["capture_id"] == 0:
                    handles = [v["shared_texture_handle"] for v in frame["views"]]
        all_handles.extend(handles)
        comparisons = []
        for replay, receipt in zip((original, held), replay_reports, strict=True):
            path = replay / f"condition-{index:02d}.npz"
            assert sha256(path) == receipt["conditions"][index]["responses_sha256"]
            with load_array(path, allow_pickle=False) as archive:
                saved = {k: archive[k] for k in archive.files}
            np.testing.assert_array_equal(saved["csp_render_clock"], source_clocks)
            # Alternative summation, with float64 before forming direction opponents.
            activity = np.maximum(saved["motion_activity"].astype(np.float64), 0)
            horizontal = (
                activity[:, :, 1] + activity[:, :, 5] - activity[:, :, 0] - activity[:, :, 4]
            )
            upward = activity[:, :, 2] + activity[:, :, 6] - activity[:, :, 3] - activity[:, :, 7]
            features = sum(
                horizontal[:, view] @ pool.templates[view, :, :, 0].T
                + upward[:, view] @ pool.templates[view, :, :, 1].T
                for view in range(4)
            )
            features = features / (4 * 721) / pool.scale
            feature_error = float(np.abs(features - saved["features"]).max())
            reference = features[saved["reference_indices"][-3:]].mean(0)
            decoded = (
                np.column_stack([(features - reference - center) / scale, np.ones(len(features))])
                @ coefficients
            )
            decode_error = float(np.abs(decoded - saved["decoded"]).max())
            assert feature_error < 1e-6 and decode_error < 1e-5
            scoring = (saved["elapsed"] >= 1.4) & (saved["elapsed"] <= 1.8)
            comparisons.append(
                dict(
                    replay=str(replay),
                    feature_error=feature_error,
                    decoded_error=decode_error,
                    mean_yaw=float(decoded[scoring, 2].mean()),
                )
            )
            if replay == original:
                white = saved["mean_luma_and_white_fraction"][:, 0].tolist()
        results.append(
            dict(
                condition=case["condition"],
                frames=len(rows),
                independent_source_handles=handles,
                readout_reconstruction=comparisons,
                original_minus_frozen_yaw=comparisons[0]["mean_yaw"] - comparisons[1]["mean_yaw"],
                source_front_luma_white_first_middle_last=[white[i] for i in (0, 30, 60)],
                unique_front_rgb_hashes=len({r["views"][0]["rgb_sha256"] for r in rows}),
            )
        )
    assert len(all_handles) == len(
        set(all_handles)
    )  # Original collector cache had no handle reuse.
    write_json(
        run / "result.json",
        dict(
            status="completed_motion_transfer_source_and_readout_audit",
            conditions=results,
            raw_archive_hashes_verified=True,
            original_clocks_exact=True,
            author_checkpoint_unchanged=True,
            calibration_refitted=False,
            original_capture_handles_unique_across_all_leases=True,
            source_arrays_or_images_modified=False,
            production_stationary_guard_bypassed=False,
            actor_admissible=False,
            physiological_transfer_qualified=False,
            goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    main()
