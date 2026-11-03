"""Actual every-observation six-target preview graph, eight-interval memory preflight only."""

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from scripts.probe_flyvis_registered_gradient import registered_features
from scripts.probe_flyvis_temporal_window import detach_exact, state_digest
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_visual_body_preview import body_features, direct_body_reference


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (plan["schema"] != "flybrain.full_preview_window_qualifier.v1"
            or plan["original_intervals"] != 64 or plan["window_intervals"] != 8
            or plan["optimizer_updates"] != 0 or plan["development_arrays_allowed"] is not False
            or plan["loss_observations"] != "every original observation after each interval"
            or plan["readout_usage"] != "ALLTRAIN frozen diagnostic only; no fold generalization"):
        raise ValueError("Sealed actual preview-graph qualifier contract required")
    for path, digest in plan["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Sealed qualifier input changed: " + path)
    data = Path(plan["registered_data"])
    source = read(data / "dataset.json")
    entry = next(e for e in source["episodes"] if e["id"] == plan["train_episode"])
    if entry["role"] != "train" or entry["id"] != "episode-04-train":
        raise ValueError("Unchanged declared TRAIN04 required")
    replay = read(Path(plan["sensor_replay"]) / "result.json")
    original = next(e for e in replay["episodes"] if e["id"] == entry["id"])
    vision = Path(original["output"])
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), original_role="train", optimizer_updates=0,
        readout_alltrain_diagnostic_only=True, fold_generalization_claimed=False))
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = dict(status="failed", optimizer_updates=0, author_parameter_changes=[],
        readout_parameter_changes=[], development_arrays_opened=0, evaluation_arrays_opened=0,
        actor_deployed=False, goal_achieved=False, fold_generalization_qualified=False,
        physiological_function_verified=False, runtime_mappings_opened=0,
        control_messages_sent=0, ui_actions_sent=0, windows=[])
    try:
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("Owned GPU qualifier required")
        os.environ["FLYVIS_ROOT_DIR"] = str(Path(plan["cache_root"]).resolve())
        os.environ["MPLBACKEND"] = "Agg"
        install_datamate_cache_compat(plan["cache_root"])
        import flyvis

        torch.manual_seed(plan["seed"])
        torch.set_default_device("cuda")
        model_dir = Path(plan["model_dir"])
        config = yaml.safe_load((model_dir / "_meta.yaml").read_text(encoding="utf8"))
        network = flyvis.Network(**config["config"]["network"]).to("cuda")
        flags = {name: p.requires_grad for name, p in network.named_parameters()}
        weights = torch.load(model_dir / "best_chkpt", map_location="cpu",
                             weights_only=True)["network"]
        network.load_state_dict(weights, strict=True)
        implementation = Path(inspect.getsourcefile(type(network)))
        write_json(run / "author-implementation.json", dict(path=str(implementation),
            sha256=sha256(implementation), original_trainable_flags=flags))
        with load_array(vision / "responses.npz", allow_pickle=False) as arrays:
            clock = arrays["decision_monotonic_seconds"].copy()
            retinas = arrays["retina_intensity"][:65].copy()
            expected = np.concatenate([arrays["additional_activity"][:65],
                                       arrays["motion_activity"][:65]], axis=2)
        if len(clock) != 739:
            raise ValueError("Complete source739 observations required")
        with load_array(vision / "retina-registration.npz", allow_pickle=False) as arrays:
            indices = np.concatenate([arrays["additional_indices"], arrays["motion_indices"]])
            rays, origins = arrays["body_directions"].copy(), arrays["body_origins_m"].copy()
        with load_array(entry["features_file"], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["observation_time"], clock)
            expected_features = arrays["registered"][:65].copy()
        with load_array(entry["targets_file"], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["observation_time"], clock)
            targets = arrays["preview"][:65].reshape(-1, 6).copy()
            valid = np.repeat(arrays["preview_valid"][:65], 2, 1)
        body, mask, _ = direct_body_reference(entry, clock, sha256(plan["body_contract"]))
        saved = torch.load(plan["readout"], map_location="cuda", weights_only=False)
        readout = RegisteredPreview().to("cuda").eval().requires_grad_(False)
        readout.load_state_dict(saved["model"])
        with load_array(plan["channel_calibration"], allow_pickle=False) as arrays:
            mean = arrays["processed_actual_mean"].copy()
            scale = arrays["processed_actual_scale"].copy()
        counts = np.zeros(6, int)
        train_ids = []
        for e in source["episodes"]:
            if e["role"] != "train":
                continue
            if sha256(e["targets_file"]) != e["targets_sha256"]:
                raise ValueError("Original TRAIN mask source changed")
            with load_array(e["targets_file"], allow_pickle=False) as arrays:
                counts += np.repeat(arrays["preview_valid"], 2, 1).sum(0)
            train_ids.append(e["id"])
        if train_ids != plan["normalization_train_ids"] or counts.tolist() != (
                plan["whole_train_valid_counts"]):
            raise ValueError("Declared allTRAIN diagnostic mask normalization changed")
        routed_body = body_features(body[:65], mask[:65], saved["body_mean"],
            saved["body_scale"], saved["plan"]["body_route_indices"])
        feature_mean = torch.tensor(saved["feature_mean"], dtype=torch.float32)
        feature_scale = torch.tensor(saved["feature_scale"], dtype=torch.float32)
        target_mean = torch.tensor(saved["target_mean"], dtype=torch.float32)
        target_scale = torch.tensor(saved["target_scale"], dtype=torch.float32)
        weight = torch.tensor(1. / (6 * counts), dtype=torch.float32)
        tensor_body = torch.tensor(routed_body)
        selected_indices = torch.tensor(indices, dtype=torch.long)
        prefix_clock, gaps = clock[:65], np.diff(clock[:65])
        if not np.all((gaps > 0) & (gaps <= .08)):
            raise ValueError("Original intervals within80ms required")
        write_json(run / "original-prefix-reference.json", dict(
            original_clocks=prefix_clock.tolist(), original_intervals=gaps.tolist(),
            original_target_masks=valid.tolist(), whole_train_valid_counts=counts.tolist(),
            target_mean=saved["target_mean"].tolist(), target_scale=saved["target_scale"].tolist(),
            per_head_loss_weights=(1. / (6 * counts)).tolist(),
            body_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            body_values_sha256=hashlib.sha256(body.tobytes()).hexdigest(),
            original_train_ids=train_ids, complete_source_observations=739,
            declared_prefix_observations=65, initial_observation_retained=True,
            readout_usage=plan["readout_usage"]))
        network.eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            state = network.steady_state(1., .005, batch_size=4, value=.5)
        state, _ = detach_exact(state)
        network.clamp()
        initialization_peak = torch.cuda.max_memory_allocated()
        if initialization_peak > plan["maximum_peak_gpu_tensor_bytes"]:
            raise ValueError("Initialization memory exceeded declared ceiling")
        torch.cuda.reset_peak_memory_stats()
        rows, windows, ticks = [], [], 0
        neural_max_error = feature_max_error = 0.

        def observation(index, state, gradient):
            nonlocal neural_max_error, feature_max_error
            selected = state.nodes.activity[:, selected_indices]
            neural_error = float(abs(selected.detach().cpu().numpy() - expected[index]).max())
            features = registered_features(selected, rays, origins, body[index:index+1],
                                           mask[index:index+1], mean, scale)
            feature_error = float(abs(features.detach().cpu().numpy()[0]
                                      - expected_features[index]).max())
            if max(neural_error, feature_error) > plan["parity_absolute_tolerance"]:
                raise ValueError("Original neuronal/registered-feature parity failed")
            neural_max_error = max(neural_max_error, neural_error)
            feature_max_error = max(feature_max_error, feature_error)
            normalized = ((features-feature_mean)/feature_scale).clamp(-5, 5)
            output = readout(normalized, tensor_body[index:index+1])
            target = torch.tensor(targets[index:index+1], dtype=torch.float32)
            available = torch.tensor(valid[index:index+1])
            losses = (output-(target-target_mean)/target_scale).square()*available*weight
            scalar = losses.sum()
            if not torch.isfinite(scalar) or scalar.requires_grad is not gradient:
                raise ValueError("Unexpected preview diagnostic gradient state")
            prediction = output.detach().cpu().numpy()[0]*saved["target_scale"]+(
                saved["target_mean"])
            row = dict(index=index, original_clock=float(prefix_clock[index]),
                held_retina_index=index-1 if index else None,
                original_elapsed_seconds=float(gaps[index-1]) if index else None,
                original_target_mask=valid[index].tolist(), prediction=prediction.tolist(),
                per_head_weighted_loss=losses.detach().cpu().numpy()[0].tolist(),
                summed_weighted_loss=float(scalar.detach()), neural_max_error=neural_error,
                feature_max_error=feature_error, contributes_gradient=gradient,
                numeric_state_hashes=state_digest(state))
            rows.append(row)
            return scalar

        with torch.no_grad():
            observation(0, state, False)
        for name, parameter in network.named_parameters():
            parameter.requires_grad_(flags[name])
        for start in range(0, 64, 8):
            wall = time.perf_counter()
            network.zero_grad(set_to_none=True)
            params = network._param_api()
            loss = torch.zeros((), dtype=torch.float32)
            for i in range(start, start+8):
                network.stimulus.zero(4, 1)
                network.stimulus.add_input(torch.tensor(retinas[i])[:, None, None, :])
                stimulus = network.stimulus()[:, 0].clone()
                intervals = integration_intervals(float(gaps[i]))
                for dt in intervals:
                    state = network._next_state(params, state, stimulus, dt)
                ticks += len(intervals)
                loss = loss + observation(i+1, state, True)
                if torch.cuda.max_memory_allocated() > plan["maximum_peak_gpu_tensor_bytes"]:
                    raise ValueError("Actual every-observation preview graph exceeded14GB")
            if not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError("Finite eight-observation differentiable loss required")
            loss.backward()
            gradients, nonzero = [], 0
            for name, parameter in network.named_parameters():
                if not flags[name] and parameter.grad is not None:
                    raise ValueError("Originally fixed author parameter received gradient")
                if flags[name]:
                    grad = parameter.grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise ValueError("Missing/nonfinite actual preview author gradient")
                    changed = int(torch.count_nonzero(grad))
                    nonzero += changed
                    gradients.append(dict(name=name, elements=parameter.numel(),
                        nonzero=changed, norm=float(grad.norm())))
            valid_counts = valid[start+1:start+9].sum(0)
            if valid_counts.sum() and not nonzero:
                raise ValueError("Valid original target window had no author gradient")
            if any(p.grad is not None for p in readout.parameters()):
                raise ValueError("Frozen diagnostic readout received gradient")
            state, leaves = detach_exact(state)
            window = dict(start=start, end=start+8, original_observation_indices=list(
                range(start+1, start+9)), gradient_target_counts=valid_counts.tolist(),
                summed_weighted_loss=float(loss.detach()), nonzero_author_gradient_values=nonzero,
                gradients=gradients, exact_detach=True, detached_state_tensors=leaves,
                state_hashes_after_detach=state_digest(state),
                wall_seconds=time.perf_counter()-wall,
                allocated_gpu_bytes=torch.cuda.memory_allocated(),
                peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated(), optimizer_updates=0)
            windows.append(window)
            result["windows"] = windows
            write_json(run / "progress.json", dict(original_intervals_covered=start+8,
                logical_internal_ticks=ticks, window=window))
            del loss, params, stimulus
        changed_author = [k for k, value in network.state_dict().items()
                          if k in weights and not torch.equal(
                              value.detach().cpu(), weights[k].cpu())]
        changed_readout = [k for k, value in readout.state_dict().items()
                           if not torch.equal(value.detach().cpu(), saved["model"][k].cpu())]
        if changed_author or changed_readout:
            raise ValueError("Qualifier changed author/readout parameter values")
        graph_peak = torch.cuda.max_memory_allocated()
        if graph_peak > plan["maximum_peak_gpu_tensor_bytes"]:
            raise ValueError("Preview graph memory ceiling exceeded")
        write_json(run / "observations.json", rows)
        result.update(status="completed_full_preview_window_qualifier",
            prefix_observations=65, original_intervals=64, logical_internal_ticks=ticks,
            elapsed_seconds=float(prefix_clock[-1]-prefix_clock[0]),
            neural_max_error=neural_max_error, feature_max_error=feature_max_error,
            initialization_peak_gpu_bytes=initialization_peak,
            every_observation_graph_peak_gpu_bytes=graph_peak,
            maximum_total_measured_gpu_bytes=max(initialization_peak, graph_peak),
            gradients_cover_original_valid_masks=True, exact_detach_boundaries=8,
            recurrent_numeric_resets=0, author_parameter_changes=changed_author,
            readout_parameter_changes=changed_readout, original_trainable_elements=734,
            fixed_parameter_elements=2959, observations_sha256=sha256(run / "observations.json"),
            scope=("Actual full preview graph memory/gradient preflight; frozen ALLTRAIN "
                   "readout diagnostic only; no fitting/generalization"))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run / "result.json", result)


if __name__ == "__main__":
    main()

