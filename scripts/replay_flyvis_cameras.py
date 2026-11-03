"""Run retained AC camera streams through independent stateful author visual networks."""

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import (
    VIEW_ORDER,
    FlyvisCameraProjector,
    integration_intervals,
)
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--body-replay", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--camera-config", default="configs/ac-csp-yaw-retina.json")
    parser.add_argument("--output", required=True)
    parser.add_argument('--fractional-frozen-tail', action='store_true',
                        help='Previously verified graph backend; keep original fractional dt')
    parser.add_argument('--retinotopic-types',
                        help='Explicit extra modeled cell types, comma separated, complete columns')
    parser.add_argument('--reference-replay',
                        help='Require original retina/motion/central responses within2e-6')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required, no CPU fallback")
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(args.cache_root).resolve())
    os.environ["MPLBACKEND"] = "Agg"
    root, model = Path(args.recording), Path(args.model_dir)
    camera_config = json.loads(Path(args.camera_config).read_text(encoding="utf8"))
    source_config = json.loads((root / "manifest.json").read_text(encoding="utf8"))["config"]
    recorded_camera = source_config.get("pixel_configuration", source_config.get(
        "sensory_configuration"))
    if (recorded_camera is None
            or recorded_camera.get("views") != camera_config["views"]
            or recorded_camera.get('camera_mount_contract') != camera_config.get(
                'camera_mount_contract')
            or recorded_camera.get("view_order") != camera_config["view_order"]):
        raise ValueError("Original camera geometry/order differs from requested projection")
    bundle = json.loads((root / "vision/bundle.json").read_text(encoding="utf8"))
    archive_path = root / "vision" / bundle["file"]
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=863009,
            checkpoint_sha256=sha256(model / "best_chkpt"),
            camera_config_sha256=sha256(args.camera_config),
            events_sha256=sha256(root / "events.jsonl"),
            body_replay_result_sha256=sha256(Path(args.body_replay) / "result.json"),
            maximum_internal_dt=0.005,
            actuator_messages_sent=0,
            optimizer_updates=0,
            temporal_policy=("causal zero-order hold of previous known retina "
                             "over original decision interval"),
            initialization=("one 1-second author grey steady state, "
                            "one batch element per independent camera"),
            optical_fidelity=("engineering normalized image lattice, "
                              "not measured 5.8-degree fly optics"),
        ),
    )
    result = dict(
        status="failed",
        actuator_messages_sent=0,
        optimizer_updates=0,
        physiological_reproduction_verified=False,
        actual_actor_integration=False,
        simulated_clock_alignment_verified=False,
        goal_achieved=False,
    )
    try:
        if sha256(archive_path) != bundle["sha256"]:
            raise ValueError("Retained RGB archive changed")
        body_root = Path(args.body_replay)
        body_receipt = json.loads((body_root / "result.json").read_text(encoding="utf8"))
        if sha256(body_root / "virtual-body.npz") != body_receipt["virtual_body_sha256"]:
            raise ValueError("Body replay changed")
        with load_array(body_root / "virtual-body.npz", allow_pickle=False) as body:
            body_times = body["decision_monotonic_seconds"].copy()
            body_packets = body["native_packet_ids"].copy()
        observations = []
        for row in (json.loads(line) for line in (root / "events.jsonl").open(encoding="utf8")):
            kind = row.get("kind")
            if kind in ("student_request", "handoff_wait_observation"):
                actor = row["actor_input"]
                observations.append(
                    (
                        kind,
                        actor["image"],
                        actor["native_packet_id"],
                        actor["decision_monotonic_seconds"],
                    )
                )
            elif kind == "expert_observation":
                observations.append(
                    (
                        kind,
                        row["camera"],
                        row["native_packet_id"],
                        row["decision_monotonic_seconds"],
                    )
                )
            elif kind is None and all(key in row for key in (
                    "native", "camera", "decision_monotonic_seconds")):
                observations.append(("engine_ai_observation", row["camera"],
                                     row["native"]["packet_id"],
                                     row["decision_monotonic_seconds"]))
        np.testing.assert_array_equal([row[3] for row in observations], body_times)
        np.testing.assert_array_equal([row[2] for row in observations], body_packets)
        install_datamate_cache_compat(args.cache_root)
        import flyvis

        torch.set_default_device("cuda")
        config = yaml.safe_load((model / "_meta.yaml").read_text(encoding="utf8"))
        network = flyvis.Network(**config["config"]["network"]).to("cuda")
        weights = torch.load(model / "best_chkpt", map_location="cpu", weights_only=True)["network"]
        network.load_state_dict(weights, strict=True)
        network.eval().requires_grad_(False)
        retina_indices = network.stimulus.input_index[0]
        nodes = network.connectome.nodes
        u, v = nodes.u[:][retina_indices], nodes.v[:][retina_indices]
        projector = FlyvisCameraProjector(
            camera_config, u, v
        )
        central = network.connectome.central_cells_index[:]
        central_types = nodes.type[:].astype(str)[central]
        motion_types = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
        all_types, all_u, all_v = nodes.type[:].astype(str), nodes.u[:], nodes.v[:]
        motion_indices = []
        for name in motion_types:
            layer = np.flatnonzero(all_types == name)
            mapping = {(int(all_u[index]), int(all_v[index])): index for index in layer}
            motion_indices.append([mapping[(int(q), int(r))] for q, r in zip(u, v, strict=True)])
        motion_indices = np.array(motion_indices)
        additional_types = args.retinotopic_types.split(',') if args.retinotopic_types else []
        extra_registration = {}
        if additional_types:
            if set(additional_types) & set(motion_types):
                raise ValueError('Additional types must be distinct from existing motion outputs')
            additional_indices = column_indices(all_types, all_u, all_v, u, v, additional_types)
            extra_registration = dict(additional_indices=additional_indices,
                                      additional_types=np.array(additional_types))
        if camera_config.get('camera_mount_contract') is not None:
            origins = np.asarray([camera_config['views'][name]['origin_body_m']
                                  for name in VIEW_ORDER], float)
            if origins.shape != (4, 3) or not np.isfinite(origins).all():
                raise ValueError('Explicit finite new virtual-eye origins required')
            extra_registration.update(body_origins_m=origins,
                camera_mount_contract_json=np.array(json.dumps(
                    camera_config['camera_mount_contract'], sort_keys=True)))
        np.savez_compressed(
            run / "retina-registration.npz",
            u=u,
            v=v,
            image_uv=projector.uv,
            body_directions=projector.directions_body,
            view_order=np.array(VIEW_ORDER),
            motion_indices=motion_indices,
            motion_types=np.array(motion_types),
            **extra_registration,
        )
        activities, motion, retinas, capture_ids, phase_names = [], [], [], [], []
        additional = []
        internal_ticks, reuse = 0, 0
        last_clock, last_capture, generation, held = None, None, None, None
        cache = {}
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with zipfile.ZipFile(archive_path) as archive, torch.no_grad():
            state = network.steady_state(1.0, 0.005, batch_size=4, value=0.5)
            if args.fractional_frozen_tail:
                from flybrain_drive.flyvis_fractional_tail import FractionalVisualTail
                network.simulate = FractionalVisualTail(network, state).simulate
            for index, (phase, image, _packet, clock) in enumerate(observations):
                if generation is None:
                    generation = image["generation"]
                if image["generation"] != generation:
                    raise ValueError("Camera generation changed; no hidden neural reset")
                capture = image["capture_id"]
                if last_capture is not None and capture < last_capture:
                    raise ValueError("Camera sequence reversed")
                if last_clock is not None:
                    for dt in integration_intervals(clock - last_clock):
                        # Integrate previous known input, without inventing camera exposures.
                        state = network.simulate(
                            held[:, None, None, :], dt=dt, initial_state=state, as_states=True
                        )[-1]
                        internal_ticks += 1
                views = {row["view_id"]: row for row in image["views"]}
                if set(views) != set(VIEW_ORDER):
                    raise ValueError("Missing camera stream; no grey/black/oracle fallback")
                projected = []
                for name in VIEW_ORDER:
                    view = views[name]
                    if not 0 <= clock - view["render_started_lower"] <= 0.15:
                        raise ValueError("Original camera exposure bound stale or from future")
                    key = (capture, name)
                    if key not in cache:
                        file = view["file"].replace("\\", "/").removeprefix("vision/")
                        raw = archive.read(file)
                        if hashlib.sha256(raw).hexdigest() != view["file_sha256"]:
                            raise ValueError("Original camera file hash mismatch")
                        rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
                        if hashlib.sha256(rgb.tobytes()).hexdigest() != view["rgb_sha256"]:
                            raise ValueError("Original RGB hash mismatch")
                        if len(cache) >= 64:
                            cache.pop(next(iter(cache)))
                        cache[key] = projector.sample(rgb)
                    projected.append(cache[key])
                retinal = np.stack(projected)
                held = torch.tensor(retinal, device="cuda")
                if not torch.isfinite(state.nodes.activity).all():
                    raise ValueError("Nonfinite neural state in continuous camera episode")
                activities.append(state.nodes.activity[:, central].cpu().numpy())
                motion.append(state.nodes.activity[:, motion_indices].cpu().numpy())
                if additional_types:
                    additional.append(state.nodes.activity[:, additional_indices].cpu().numpy())
                retinas.append(retinal)
                capture_ids.append(capture)
                phase_names.append(phase)
                reuse += int(last_capture == capture)
                last_clock, last_capture = clock, capture
                if index % 50 == 0:
                    write_json(
                        run / "progress.json",
                        dict(
                            completed=index + 1,
                            total=len(observations),
                            internal_ticks=internal_ticks,
                            elapsed_seconds=time.perf_counter() - started,
                        ),
                    )
        torch.cuda.synchronize()
        changed = [
            key
            for key, value in network.state_dict().items()
            if key in weights and not torch.equal(value.cpu(), weights[key].cpu())
        ]
        if changed:
            raise ValueError("Author weights changed")
        baseline_errors = None
        if args.reference_replay:
            reference = Path(args.reference_replay)
            original = json.loads((reference/'result.json').read_text(encoding='utf8'))
            if (original['status'] != 'completed'
                    or sha256(reference/'responses.npz') != original['responses_sha256']
                    or json.loads((reference/'manifest.json').read_text('utf8'))['config'][
                        'events_sha256'] != sha256(root/'events.jsonl')):
                raise ValueError('Unchanged same-history reference replay required')
            baseline_errors = {}
            with load_array(reference/'responses.npz', allow_pickle=False) as baseline:
                np.testing.assert_array_equal(body_times, baseline['decision_monotonic_seconds'])
                for name, values in [('central_activity', activities), ('motion_activity', motion),
                                     ('retina_intensity', retinas)]:
                    error = float(np.abs(np.stack(values)-baseline[name]).max())
                    baseline_errors[name] = error
                    if error > 2e-6:
                        failed = run/'mismatched-original-responses.npz'
                        np.savez_compressed(failed, central_activity=np.stack(activities),
                            motion_activity=np.stack(motion), retina_intensity=np.stack(retinas),
                            decision_monotonic_seconds=body_times)
                        result.update(reference_replay_max_errors=baseline_errors,
                                      mismatch_arrays_sha256=sha256(failed))
                        raise ValueError('Original neural replay differs after readout expansion')
        np.savez_compressed(
            run / "responses.npz",
            central_activity=np.stack(activities),
            motion_activity=np.stack(motion),
            retina_intensity=np.stack(retinas),
            decision_monotonic_seconds=body_times,
            capture_ids=np.array(capture_ids),
            phases=np.array(phase_names),
            central_cell_types=central_types,
            **(dict(additional_activity=np.stack(additional),
                    additional_cell_types=np.array(additional_types)) if additional_types else {}),
        )
        result.update(
            status="completed",
            termination_reason="complete original camera/body episode replay",
            observations=len(observations),
            independent_camera_states=4,
            internal_ticks=internal_ticks,
            repeated_capture_decisions=reuse,
            neural_resets_at_handoff=0,
            original_elapsed_seconds=float(body_times[-1] - body_times[0]),
            seconds=time.perf_counter() - started,
            peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated(),
            weights_changed=[],
            fractional_frozen_tail=args.fractional_frozen_tail,
            additional_retinotopic_types=additional_types,
            reference_replay_max_errors=baseline_errors,
            responses_sha256=sha256(run / "responses.npz"),
            all_retinal_nodes_have_source_pixels=True,
            future_frame_used_for_previous_interval=False,
            known_limitations=[
                "T4d contrast mismatch in original model",
                "camera-specific optical scale is engineering resampling",
                "CSP exposure clocks not simulation-aligned",
                "four independent visual states, no flight-network fusion/control yet",
            ],
        )
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run / "result.json", result)
    if result["status"] != "completed":
        raise RuntimeError(result["reason"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
