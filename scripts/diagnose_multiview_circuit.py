"""Render four real MuJoCo perspectives and exercise a shared fly-structure circuit.

The scenario uses fixed diagnostic controls. Model outputs are recorded but never
applied. Mirror-named views are direct virtual cameras, not native AC reflections.
"""

import argparse
import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.compound_eye import PerspectiveView
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController, MultiViewCircuitRunner
from flybrain_drive.multiview_sensory import MultiViewPacker, ViewMap
from flybrain_drive.sensor_contract import ActorObservation, SampleTime, VisualViewSample
from scripts.initialize_vehicle_circuit import organs


def save_raw_png(path, rgb):
    """Lossless raw RGB artifact without an optional imaging-library dependency."""
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Raw PNG requires uint8 H x W x RGB")

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    header = struct.pack(">IIBBBBB", pixels.shape[1], pixels.shape[0], 8, 2, 0, 0, 0)
    rows = b"".join(b"\0" + row.tobytes() for row in pixels)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiview-diagnostic.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    if (type(config["decisions"]) is not int or not 2 <= config["decisions"] <= 1000
            or len(config["views"]) != 4
            or any(type(v["period_decisions"]) is not int or v["period_decisions"] < 1
                   for v in config["views"])):
        raise ValueError("Bounded decisions and four explicitly scheduled views required")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    run = begin_run(args.output, {"configuration": config, "model_used_for_actuation": False,
                                 "scope": "MuJoCo multi-view component diagnostic"})
    env, renderer = None, None
    try:
        from flybrain_drive.env import FlyBrainEnv
        from flybrain_drive.mujoco_multiview import MuJoCoMultiViewRenderer
        from flybrain_drive.mujoco_sensory import MuJoCoSensoryBridge

        if sha256(config["source_checkpoint"]) != config["source_checkpoint_sha256"]:
            raise ValueError("Source initialization checkpoint hash differs")
        source = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=True)
        if source.get("model_kind") != "vehicle_signed_rate_v1":
            raise ValueError("Explicit signed rate source model required")
        graph = {k: np.asarray(v, dtype={"pre": np.int64, "post": np.int64,
                                        "weights": np.float32}.get(k))
                 for k, v in source["graph"].items()}
        old = source["constructor"]
        source_config = json.loads(Path(config["source_config"]).read_text(encoding="utf8"))
        environment = {**source_config["environment"], "max_steps": config["decisions"] + 5}
        env = FlyBrainEnv(**environment)
        observation, _ = env.reset(seed=config["seed"])
        body_bridge = MuJoCoSensoryBridge(env, rows=2, columns=3)
        settings = source_config["sensory_preprocessing"]
        scalar_specs = json.loads(Path(config["scalar_calibration"]).read_text(
            encoding="utf8"))["scalar_organs"]
        encoder = organs(settings, scalar_specs)
        axes = np.asarray(old["directions"]).reshape(-1, 3)
        axes = axes[np.linalg.norm(axes, axis=1) > .9]
        views = []
        pitch = np.deg2rad(env.camera.pitch_degrees)
        for item in config["views"]:
            theta = np.deg2rad(item["yaw_degrees"])
            forward = [np.cos(pitch) * np.cos(theta), np.cos(pitch) * np.sin(theta), np.sin(pitch)]
            left = [-np.sin(theta), np.cos(theta), 0]
            up = [-np.sin(pitch) * np.cos(theta), -np.sin(pitch) * np.sin(theta), np.cos(pitch)]
            perspective = PerspectiveView(np.column_stack([forward, left, up]),
                item["origin_body_m"], config["fovy_degrees"])
            views.append(ViewMap(item["source_id"], axes, np.ones(len(axes)), perspective,
                "reconstructed source axes; virtual direct head camera; not an AC mirror"))
        packer = MultiViewPacker(views, encoder, max_view_age=config["max_view_age_seconds"],
                                 contrast_scale=config["temporal_RGB_contrast_scale"])
        available = []
        size = config["source_size"]
        projectors = {v.source_id: v.projector() for v in views}
        for view in views:
            _, valid, _ = projectors[view.source_id].sample_pixels(np.zeros((size, size, 3)))
            available.append(valid)
        port_sources = np.arange(len(old["visual_ids"])) % 4
        np.random.default_rng(config["seed"]).shuffle(port_sources)
        constructor = {key: old[key] for key in ("visual_ids", "organ_groups", "signs",
            "incoming_totals", "output_groups", "steering_signs", "dt", "tau")}
        constructor.update(directions=[v.directions_body.tolist() for v in views],
            available=np.asarray(available).tolist(), port_sources=port_sources.tolist(),
            sensory_calibration=packer.calibration, neighbors=config["neighbors"],
            seed=config["seed"])
        model = MultiViewCircuitController(graph, **constructor)
        with torch.no_grad():
            for i, _ in enumerate(model.adapter.group_names):
                model.adapter.organ_adapters[i].weight.copy_(
                    source["model"][f"adapter.organ_adapters.{i}.weight"])
            model.readout.load_state_dict({k.removeprefix("readout."): v
                for k, v in source["model"].items() if k.startswith("readout.")})
        runner = MultiViewCircuitRunner(model, packer)
        renderer = MuJoCoMultiViewRenderer(env, views, source_size=size)
        camera = renderer.camera
        original_camera = [env.model.cam_pos[camera].copy(), env.model.cam_quat[camera].copy(),
                           np.array(env.model.cam_fovy[camera]).copy()]
        cached, images, frame_ids = {}, {}, {v.source_id: -1 for v in views}
        trace, rgb_frames, first_packed = [], [], None
        for tick in range(config["decisions"]):
            if tick:
                observation, _, terminated, truncated, _ = env.step(config["scenario_control"])
                if terminated or truncated:
                    raise ValueError("Diagnostic scenario ended before its declared budget")
            clock = float(env.data.time)
            scheduled = [v["source_id"] for v in config["views"]
                         if tick % v["period_decisions"] == 0]
            fresh = renderer.render(scheduled)
            for name, pixels in fresh.items():
                view = views[[v.source_id for v in views].index(name)]
                response, valid, _ = projectors[name].sample_pixels(pixels)
                frame_ids[name] += 1
                cached[name] = VisualViewSample(name, frame_ids[name], response, valid,
                    SampleTime(clock, time.monotonic()), view.sha256, "declared_sensor_model")
                images[name] = pixels
            legacy = body_bridge.observe(env, observation, episode_id="multiview-component",
                sequence_id=tick, received_monotonic_seconds=time.monotonic())
            drop = config["drop_source"]
            present = [v.source_id for v in views if not (v.source_id == drop["source_id"]
                       and drop["start_decision"] <= tick < drop["end_decision_exclusive"])]
            actor = ActorObservation("multiview-component", tick, clock, time.monotonic(), (),
                                     legacy.body, views=tuple(cached[name] for name in present))
            requests, packed = runner.step(actor)
            if (not np.isfinite(requests).all() or np.any(requests < [-1, 0, 0])
                    or np.any(requests > 1)):
                raise ValueError("Invalid model request range")
            if first_packed is None:
                first_packed = {k: torch.tensor(v)[None] for k, v in packed.items()}
            trace.append({"tick": tick, "simulation_seconds": clock,
                "view_frame_ids": [cached[v.source_id].frame_id if v.source_id in present else -1
                                   for v in views],
                "view_exposure_simulation_seconds": [cached[v.source_id].time.simulation_seconds
                                                       for v in views],
                "view_age": packed["view_age"].tolist(),
                "valid_rays": packed["view_valid"].sum(1).astype(int).tolist(),
                "valid_temporal_contrast_rays": packed["contrast_valid"].sum(1).astype(
                    int).tolist(),
                "model_request_not_applied": requests.tolist()})
            rgb_frames.append(np.stack([images[v.source_id] for v in views]))
        for before, after in zip(original_camera, [env.model.cam_pos[camera],
            env.model.cam_quat[camera], env.model.cam_fovy[camera]], strict=True):
            np.testing.assert_array_equal(before, after)
        np.savez_compressed(run / "raw-direct-views.npz", rgb=np.stack(rgb_frames),
                            source_ids=np.array([v.source_id for v in views]))
        for i, view in enumerate(views):
            save_raw_png(run / (view.source_id + ".png"), rgb_frames[-1][i])
        with torch.no_grad():
            baseline = model.prepare_drive(first_packed)
            isolation = []
            for i in range(4):
                changed = {k: v.clone() for k, v in first_packed.items()}
                changed["view_rgb"][:, i] = (changed["view_rgb"][:, i] + .1).clamp(0, 1)
                delta = model.prepare_drive(changed) - baseline
                allowed = torch.zeros(model.circuit.size, dtype=torch.bool)
                allowed[model.adapter.visual_ports[port_sources == i]] = True
                error = float(delta[:, ~allowed].abs().max())
                if error != 0:
                    raise ValueError("A local source perturbation reached another input group")
                isolation.append(error)
            states = []
            for selected in ([0], [1], [2], [3], [0, 1, 2, 3]):
                values = {k: v.clone() for k, v in first_packed.items()}
                values["organ_valid"].zero_()
                values["view_valid"].zero_()
                values["view_valid"][:, selected] = first_packed["view_valid"][:, selected]
                state = model.initial_state()
                drive = model.prepare_drive(values)
                for _ in range(64):
                    _, state = model.advance(drive, state)
                states.append(state)
            internal = torch.tensor(graph["roles"] == "internal")
            interaction = float((states[4] - sum(states[:4]))[:, internal].abs().sum())
            common = int(torch.stack([s[:, internal] > 0 for s in states[:4]]).all(0).sum())
        payload = {"model_kind": "multiview_vehicle_rate_v1", "constructor": constructor,
            "graph": {k: v.tolist() for k, v in graph.items()}, "model": model.state_dict(),
            "configuration": config, "optimizer_updates": 0, "driving_trained": False}
        torch.save(payload, run / "multiview-initialized.pt")
        restored = MultiViewCircuitController(graph, **constructor)
        restored.load_state_dict(torch.load(run / "multiview-initialized.pt",
                                            weights_only=True)["model"])
        a, b = model.initial_state(), restored.initial_state()
        with torch.no_grad():
            for _ in range(12):
                x, a = model(first_packed, a)
                y, b = restored(first_packed, b)
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        write_json(run / "trace.json", trace)
        write_json(run / "view-maps.json", [v.manifest() for v in views])
        result = {"status": "completed", "nodes": len(graph["ids"]), "views": 4,
            "decisions": len(trace), "reconstructed_axes_per_source": len(axes),
            "valid_rays_per_source": np.sum(available, axis=1).astype(int).tolist(),
            "visual_ports_per_source": np.bincount(port_sources).tolist(),
            "source_isolation_max_errors": isolation, "camera_parameters_restored": True,
            "internal_joint_vs_independent_state_l1": interaction,
            "internal_neurons_active_from_every_individual_source": common,
            "checkpoint_restore_max_error": 0,
            "checkpoint_sha256": sha256(run / "multiview-initialized.pt"),
            "raw_images_sha256": sha256(run / "raw-direct-views.npz"),
            "raw_source": "MuJoCo rendered direct perspectives; not physical or AC mirrors",
            "visual_transduction_reinitialized": True, "driving_trained": False,
            "optimizer_updates": 0, "model_used_for_actuation": False,
            "fusion_efficiency_advantage_verified": False, "AC_driving_verified": False,
            "goal_achieved": False}
        write_json(run / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error), "goal_achieved": False})
        raise
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
