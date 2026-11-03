"""Fixed author network on diagnostic-only actual AC camera rotation videos."""

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from flybrain_drive.ac_csp_geometry import camera_geometry, matrix
from flybrain_drive.ac_csp_vision import validate_frame_aid_state
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER, FlyvisCameraProjector
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def diagnostic_steps(elapsed):
    # All these frames are rejected by actor admission. Preserve even a diagnostic
    # camera gap by holding its last image; flag >150ms instead of editing clocks.
    if not np.isfinite(elapsed) or not 0 < elapsed <= 8:
        raise ValueError("Positive bounded original diagnostic interval required")
    full = int(np.floor(elapsed / 0.005))
    result = [0.005] * full
    remainder = elapsed - 0.005 * full
    if remainder > 1e-12:
        result.append(remainder)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--calibration", default="runs/visual-self-motion-reference-v2-20261029-t1032"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--hold-first-retina",
        action="store_true",
        help="Counterfactual: keep first source retina; clocks/body/targets unchanged",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(1)
    torch.manual_seed(863379)
    os.environ["FLYVIS_ROOT_DIR"] = args.cache_root
    install_datamate_cache_compat(args.cache_root)
    import flyvis

    torch.set_default_device("cuda")
    root, model, calibration = Path(args.collection), Path(args.model_dir), Path(args.calibration)
    collection = read(root / "result.json")
    assert collection["status"] == "completed_paused_camera_rotation_collection"
    assert len(collection["conditions"]) == 6
    calibrated = read(calibration / "result.json")
    assert calibrated["calibration_gate_passed"]
    assert sha256(calibration / "calibration.npz") == calibrated["calibration_sha256"]
    with load_array(calibration / "calibration.npz", allow_pickle=False) as archive:
        center, scale, coefficient = (
            archive[k].copy() for k in ("center", "scale", "coefficients")
        )
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            arguments=vars(args),
            script_sha256=sha256(__file__),
            collection_result_sha256=sha256(root / "result.json"),
            author_checkpoint_sha256=sha256(model / "best_chkpt"),
            fixed_synthetic_calibration_sha256=calibrated["calibration_sha256"],
            reference_policy="Observed paused static camera prefix; offline transfer only",
            motion_scoring_window_elapsed_seconds=[1.4, 1.8],
            diagnostic_gaps_held_at_original_times_never_actor_admitted=True,
            optimizer_updates=0,
            ui_actions_sent=0,
            actuator_messages_sent=0,
        ),
    )
    config = yaml.safe_load((model / "_meta.yaml").read_text(encoding="utf8"))
    network = flyvis.Network(**config["config"]["network"]).to("cuda")
    weights = torch.load(model / "best_chkpt", map_location="cpu", weights_only=True)["network"]
    network.load_state_dict(weights, strict=True)
    network.eval().requires_grad_(False)
    nodes = network.connectome.nodes
    indices = network.stimulus.input_index[0]
    u, v = nodes.u[:][indices], nodes.v[:][indices]
    camera = read("configs/ac-csp-yaw-retina.json")
    assert (
        sha256("configs/ac-csp-yaw-retina.json")
        == read(calibration / "manifest.json")["config"]["camera_config_sha256"]
    )
    projector = FlyvisCameraProjector(camera, u, v)
    pool = VisualSelfMotionPool(camera, projector.directions_body)
    types, all_u, all_v = nodes.type[:].astype(str), nodes.u[:], nodes.v[:]
    motion_indices = []
    for name in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
        layer = np.flatnonzero(types == name)
        mapping = {(int(all_u[i]), int(all_v[i])): i for i in layer}
        motion_indices.append([mapping[(int(q), int(r))] for q, r in zip(u, v, strict=True)])
    motion_indices = np.asarray(motion_indices)
    reports = []
    with torch.no_grad():
        for entry in collection["conditions"]:
            source = Path(entry["output"])
            assert sha256(source / "result.json") == entry["result_sha256"]
            rows = read(source / "rows.json")
            bundle = read(source / "vision/bundle.json")
            assert sha256(source / "vision/camera.zip") == bundle["sha256"]
            before, after = read(source / "before.json"), read(source / "after.json")
            assert before["session"]["paused"] and after["session"]["paused"]
            assert before["session"]["generation"] == after["session"]["generation"]
            clocks, elapsed, angles, features, activity, luma, baseline_looks = (
                [],
                [],
                [],
                [],
                [],
                [],
                {},
            )
            held, first_retina, last, ticks, pose_error = None, None, None, 0, 0.0
            state = network.steady_state(1.0, 0.005, batch_size=4, value=0.5)
            with zipfile.ZipFile(source / "vision/camera.zip") as archive:
                for row in rows:
                    raw = archive.read(row["file"])
                    assert hashlib.sha256(raw).hexdigest() == row["file_sha256"]
                    frame = json.loads(raw)
                    assert frame["ideal_line_shown"] is False and frame["aid_probe_only"] is True
                    try:
                        validate_frame_aid_state(frame)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Diagnostic frame must not be actor admissible")
                    camera_geometry(frame["views"])
                    diagnostic = frame["camera_motion_diagnostic"]
                    assert diagnostic["physical_body_paused"] is True
                    assert (
                        diagnostic["requested_yaw_left_rad_s"]
                        == entry["condition"]["yaw_left_rad_s"]
                    )
                    time = frame["views"][0]["render_started_clock"]
                    if last is not None:
                        for dt in diagnostic_steps(time - last):
                            state = network.simulate(
                                held[:, None, None, :], dt=dt, initial_state=state, as_states=True
                            )[-1]
                            ticks += 1
                    pixels = []
                    views = {r["view_id"]: r for r in row["views"]}
                    exposure = []
                    for view in frame["views"]:
                        name = view["view_id"]
                        raw = archive.read(views[name]["file"])
                        assert hashlib.sha256(raw).hexdigest() == views[name]["file_sha256"]
                        rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
                        assert (
                            hashlib.sha256(rgb.tobytes()).hexdigest() == views[name]["rgb_sha256"]
                        )
                        pixels.append((name, projector.sample(rgb)))
                        exposure.append([float(rgb.mean() / 255), float((rgb >= 250).mean())])
                        rotation = matrix(view["view_matrix_diagnostic"])[:3, :3]
                        look, up = -rotation[:, 2], rotation[:, 1]
                        if name not in baseline_looks:
                            baseline_looks[name] = look
                        angle = np.arctan2(
                            up @ np.cross(baseline_looks[name], look), baseline_looks[name] @ look
                        )
                        pose_error = max(
                            pose_error, abs(float(angle) - diagnostic["yaw_left_angle_rad"])
                        )
                    current_retina = torch.tensor(
                        np.stack([dict(pixels)[name] for name in VIEW_ORDER])
                    )
                    if first_retina is None:
                        first_retina = current_retina.clone()
                    held = first_retina if args.hold_first_retina else current_retina
                    motion = state.nodes.activity[:, motion_indices].cpu().numpy()
                    assert np.isfinite(motion).all()
                    features.append(pool.features(motion))
                    activity.append(motion)
                    clocks.append(time)
                    elapsed.append(diagnostic["elapsed_csp_seconds"])
                    angles.append(diagnostic["yaw_left_angle_rad"])
                    luma.append(exposure)
                    last = time
            assert pose_error < 2e-5
            clocks, elapsed, angles, features = map(np.asarray, (clocks, elapsed, angles, features))
            reference_rows = np.flatnonzero((elapsed < 0.6) & (np.abs(angles) < 1e-12))
            assert len(reference_rows) >= 11
            assert clocks[reference_rows[-1]] - clocks[reference_rows[0]] >= 0.4
            reference = features[reference_rows[-3:]].mean(0)
            design = np.column_stack(
                [(features - reference - center) / scale, np.ones(len(features))]
            )
            decoded = design @ coefficient
            selected = (elapsed >= 1.4) & (elapsed <= 1.8)
            assert selected.sum() >= 4
            average = decoded[selected].mean(0)
            output = run / f"condition-{entry['condition']['index']:02d}.npz"
            np.savez_compressed(
                output,
                csp_render_clock=clocks,
                elapsed=elapsed,
                angles=angles,
                motion_activity=np.asarray(activity),
                features=features,
                decoded=decoded,
                mean_luma_and_white_fraction=np.asarray(luma),
                reference_indices=reference_rows,
            )
            reports.append(
                dict(
                    condition=entry["condition"],
                    original_frames=len(rows),
                    pose_angle_max_error_rad=pose_error,
                    internal_ticks=ticks,
                    original_intervals_above150ms=int((np.diff(clocks) > 0.15).sum()),
                    maximum_original_interval_seconds=float(np.diff(clocks).max()),
                    static_prefix_observations=len(reference_rows),
                    scoring_observations=int(selected.sum()),
                    mean_six_coordinates=average.tolist(),
                    mean_yaw_rad_s=float(average[2]),
                    maximum_non_yaw_coordinate=float(np.abs(average[[0, 1, 3, 4, 5]]).max()),
                    responses_sha256=sha256(output),
                )
            )
            write_json(run / "progress.json", dict(completed=len(reports), reports=reports))
    changed = [
        k
        for k, value in network.state_dict().items()
        if k in weights and not torch.equal(value.cpu(), weights[k].cpu())
    ]
    assert not changed
    write_json(
        run / "result.json",
        dict(
            status="completed_fixed_author_camera_motion_replay",
            conditions=reports,
            author_weights_changed=changed,
            original_clocks_preserved=True,
            offline_paused_prefix_not_production_stationary_reference=True,
            neural_input_intervention=(
                "hold_first_retina_counterfactual" if args.hold_first_retina else "original_video"
            ),
            actor_admissible=False,
            actor_freshness_guards_changed=False,
            new_fly_network_weights_trained=False,
            physiological_transfer_qualified=False,
            goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    main()
