"""Calibrate a tiny geometric motion readout from controlled camera videos, no driving."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import SphereCameraStimulus, VisualSelfMotionPool


def score(prediction, target, amplitudes):
    amplitudes = np.broadcast_to(amplitudes, target.shape)
    nonzero = np.linalg.norm(target, axis=1) > 0
    rows = np.flatnonzero(nonzero)
    axes = np.abs(target[rows] / amplitudes[rows]).argmax(axis=1)
    sign_accuracy = np.mean(np.sign(prediction[rows, axes]) == np.sign(target[rows, axes]))
    dominant = np.mean(np.abs(prediction[rows] / amplitudes[rows]).argmax(axis=1) == axes)
    return dict(
        nonzero_case_sign_accuracy=float(sign_accuracy),
        dominant_axis_accuracy=float(dominant),
        normalized_rmse=float(np.sqrt(np.mean(((prediction - target) / amplitudes) ** 2))),
        static_normalized_error=float(
            np.linalg.norm(prediction[~nonzero] / amplitudes[~nonzero], axis=1).max()
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/visual-self-motion-calibration-20261029.json")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    model = Path(args.model_dir)
    torch.set_num_threads(1)
    torch.manual_seed(863033)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(args.cache_root).resolve())
    os.environ["MPLBACKEND"] = "Agg"
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            plan=config,
            seed=863033,
            config_sha256=sha256(args.config),
            checkpoint_sha256=sha256(model / "best_chkpt"),
            camera_config_sha256=sha256(config["camera_config"]),
            actuator_messages_sent=0,
        ),
    )
    result = dict(
        status="failed",
        actuator_messages_sent=0,
        optimizer_updates=0,
        author_network_weights_trained=False,
        physiological_reproduction_verified=False,
        actual_actor_integration=False,
        goal_achieved=False,
    )
    try:
        install_datamate_cache_compat(args.cache_root)
        import flyvis

        torch.set_default_device("cuda")
        network_config = yaml.safe_load((model / "_meta.yaml").read_text(encoding="utf8"))
        network = flyvis.Network(**network_config["config"]["network"]).to("cuda")
        weights = torch.load(model / "best_chkpt", map_location="cpu", weights_only=True)["network"]
        network.load_state_dict(weights, strict=True)
        network.eval().requires_grad_(False)
        nodes = network.connectome.nodes
        input_indices = network.stimulus.input_index[0]
        camera = json.loads(Path(config["camera_config"]).read_text(encoding="utf8"))
        projector = FlyvisCameraProjector(
            camera, nodes.u[:][input_indices], nodes.v[:][input_indices]
        )
        pool = VisualSelfMotionPool(camera, projector.directions_body)
        scene = SphereCameraStimulus(
            camera, config["camera_origins_body_m"], radius=config["sphere_radius_m"]
        )
        types, u, v = nodes.type[:].astype(str), nodes.u[:], nodes.v[:]
        motion_indices = []
        for name in ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]:
            lookup = {
                (int(u[index]), int(v[index])): index for index in np.flatnonzero(types == name)
            }
            motion_indices.append(
                [
                    lookup[(int(q), int(r))]
                    for q, r in zip(u[input_indices], v[input_indices], strict=True)
                ]
            )
        motion_indices = np.array(motion_indices)
        features, targets, protocols, feature_time, directional = [], [], [], [], []
        texture_seeds, all_amplitudes = [], []
        ticks = int(round(config["frame_dt"] / 0.005))
        if abs(ticks * 0.005 - config["frame_dt"]) > 1e-12:
            raise ValueError("Stimulus timing must be an exact declared internal-step multiple")
        start = time.perf_counter()
        schedules = [
            (protocol, settings)
            for protocol in ("train", "validation")
            for settings in (
                config[protocol] if isinstance(config[protocol], list) else [config[protocol]]
            )
        ]
        for protocol, settings in schedules:
            cases = [np.zeros(6)]
            amplitudes = np.array(
                [settings["angular_amplitude"]] * 3
                + [settings["normalized_translation_amplitude"]] * 3
            )
            cases += [axis * sign * amplitudes for axis in np.eye(6) for sign in (-1, 1)]
            for case_index, target in enumerate(cases):
                images = []
                for frame in range(config["frames"]):
                    t = max(0.0, frame * config["frame_dt"] - config["static_pre_seconds"])
                    rgb = scene.render(
                        target[:3], target[3:], t, texture_seed=settings["texture_seed"]
                    )
                    images.append(np.stack([projector.sample(view) for view in rgb]))
                retina = np.stack(images)  # measured stimulus frames, view, receptor
                movie = torch.tensor(
                    np.transpose(np.repeat(retina, ticks, axis=0), (1, 0, 2))[:, :, None, :],
                    device="cuda",
                )
                with torch.no_grad():
                    activity = network.simulate(movie, dt=0.005, initial_state="auto")
                    if not torch.isfinite(activity).all():
                        raise ValueError("Nonfinite visual response to declared component stimulus")
                    motion = activity[:, ticks - 1 :: ticks, motion_indices].permute(1, 0, 2, 3)
                    response = motion.cpu().numpy()
                    del activity, motion, movie
                pooled = pool.features(response)
                feature = pooled[-config["average_last_frames"] :].mean(axis=0)
                if "reference_baseline_frames" in config:
                    first, stop = config["reference_baseline_frames"]
                    if first < 0 or stop > config["frames"] or first >= stop:
                        raise ValueError("Invalid stationary reference window")
                    if (stop - 1) * config["frame_dt"] > config["static_pre_seconds"]:
                        raise ValueError("Baseline would include future moving frames")
                    feature = feature - pooled[first:stop].mean(axis=0)
                features.append(feature)
                feature_time.append(pooled)
                directional.append(np.maximum(response, 0).mean(axis=(0, 3)))
                targets.append(target)
                protocols.append(protocol)
                texture_seeds.append(settings["texture_seed"])
                all_amplitudes.append(amplitudes)
                write_json(
                    run / "progress.json",
                    dict(
                        protocol=protocol,
                        case=case_index,
                        completed=len(features),
                        total=13 * len(schedules),
                        elapsed_seconds=time.perf_counter() - start,
                    ),
                )
        features, targets = np.array(features), np.array(targets)
        train = np.array(protocols) == "train"
        center, scale = features[train].mean(axis=0), features[train].std(axis=0)
        if np.any(scale < 1e-9):
            raise ValueError("Motion feature rank/variation insufficient")
        design = np.column_stack([(features - center) / scale, np.ones(len(features))])
        ridge = np.eye(7) * config["ridge"]
        ridge[-1, -1] = 0
        calibration = np.linalg.solve(
            design[train].T @ design[train] + ridge, design[train].T @ targets[train]
        )
        predicted = design @ calibration
        all_amplitudes = np.array(all_amplitudes)
        texture_seeds = np.array(texture_seeds)
        metrics = score(predicted[~train], targets[~train], all_amplitudes[~train])
        per_texture = {}
        gates = config["gates"]

        def passes(metric):
            return (
                metric["nonzero_case_sign_accuracy"] >= gates["nonzero_case_sign_accuracy_min"]
                and metric["dominant_axis_accuracy"] >= gates["dominant_axis_accuracy_min"]
                and metric["normalized_rmse"] <= gates["normalized_rmse_max"]
                and metric["static_normalized_error"]
                <= gates.get("static_normalized_error_max", float("inf"))
            )

        for seed in np.unique(texture_seeds[~train]):
            selected = (~train) & (texture_seeds == seed)
            per_texture[str(seed)] = score(
                predicted[selected], targets[selected], all_amplitudes[selected]
            )
        passed = passes(metrics) and all(passes(metric) for metric in per_texture.values())
        np.savez_compressed(
            run / "calibration.npz",
            center=center,
            scale=scale,
            coefficients=calibration,
            templates=pool.templates,
        )
        np.savez_compressed(
            run / "responses.npz",
            features=features,
            targets=targets,
            prediction=predicted,
            protocols=np.array(protocols),
            feature_time=np.array(feature_time),
            mean_directional_activity=np.array(directional),
            texture_seeds=texture_seeds,
            amplitudes=all_amplitudes,
        )
        changed = [
            key
            for key, value in network.state_dict().items()
            if key in weights and not torch.equal(value.cpu(), weights[key].cpu())
        ]
        if changed:
            raise ValueError("Original author network weights changed")
        result.update(
            status="completed",
            termination_reason="declared calibration and held-out test completed",
            cases=len(features),
            frames_per_case=config["frames"],
            metrics=metrics,
            metrics_by_validation_texture=per_texture,
            calibration_gate_passed=passed,
            learned_parameters=42,
            training_method="closed-form ridge, no neural optimizer",
            original_network_weight_changes=[],
            seconds=time.perf_counter() - start,
            calibration_sha256=sha256(run / "calibration.npz"),
            responses_sha256=sha256(run / "responses.npz"),
            scope=(
                "known synthetic camera motion only; "
                "no road recognition or flight circuit control claim"
            ),
        )
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run / "result.json", result)
    if result["status"] != "completed":
        raise RuntimeError(result["reason"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
