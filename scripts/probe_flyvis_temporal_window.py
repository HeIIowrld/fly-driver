"""TRAIN-only author-network temporal credit/memory qualifier; no optimizer or controller."""

import argparse
import gc
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.checkpoint import checkpoint

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def state_tensors(value, prefix="state"):
    """Inspect stored state without populating derived reference caches."""
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        leaves = {}
        for key in value:
            leaves.update(state_tensors(dict.__getitem__(value, key), prefix + "." + str(key)))
        return leaves
    if hasattr(value, "values") and hasattr(value, "indices"):
        return {prefix + ".values": value.values, prefix + ".indices": value.indices}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {}
    raise TypeError("Uninspected recurrent state: " + str(type(value)))


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_digest(state):
    return {key: tensor_digest(value) for key, value in state_tensors(state).items()}


def detach_exact(state):
    before = state_digest(state)
    detached = state.detach()
    if not before or before != state_digest(detached):
        raise ValueError("Detach changed numerical state")
    if any(value.requires_grad for value in state_tensors(detached).values()):
        raise ValueError("Detach retained gradient history")
    return detached, len(before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    expected_arms = [
        {"name": "detach2_ordinary", "detach_intervals": 2, "recompute": False},
        {"name": "detach2_recompute", "detach_intervals": 2, "recompute": True},
        {"name": "detach8_recompute", "detach_intervals": 8, "recompute": True},
    ]
    if plan.get("revision") == 3:
        expected_arms = [
            {"name": "detach2_ordinary", "detach_intervals": 2, "recompute": False},
            {"name": "detach8_ordinary", "detach_intervals": 8, "recompute": False},
        ]
    if plan.get("revision") == 4:
        expected_arms = [
            {"name": "detach2_ordinary", "detach_intervals": 2, "recompute": False},
            {"name": "detach2_repeat", "detach_intervals": 2, "recompute": False},
            {"name": "detach8_ordinary", "detach_intervals": 8, "recompute": False},
        ]
    if (plan["schema"] != "flybrain.author_temporal_window_qualifier.v1"
            or plan["arms"] != expected_arms or plan["optimizer_window_intervals"] != 8
            or plan["original_intervals"] != 64 or plan["optimizer_updates"] != 0
            or plan["diagnostic_objective"] != "fixed weighted mean squared local neuron activity"
            or plan["development_arrays_allowed"] is not False):
        raise ValueError("Explicit unchanged TRAIN-only 64-interval qualifier required")
    for path, digest in plan["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Sealed qualifier input changed: " + path)
    sensor = read(Path(plan["sensor_replay"]) / "result.json")
    entry = next(e for e in sensor["episodes"] if e["id"] == plan["train_episode"])
    if entry["role"] != "train" or entry["id"] != "episode-04-train":
        raise ValueError("One declared unchanged original TRAIN history required")
    vision = Path(entry["output"])
    if sha256(vision / "responses.npz") != entry["responses_sha256"]:
        raise ValueError("Original TRAIN neuronal replay changed")
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), episode=entry["id"], original_role=entry["role"],
        no_target_or_readout_arrays=True, optimizer_updates=0, ui_actions_sent=0))
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = dict(status="failed", optimizer_updates=0, readout_arrays_opened=0,
        target_arrays_opened=0, development_arrays_opened=0, evaluation_arrays_opened=0,
        actor_deployed=False, goal_achieved=False, physiological_function_verified=False,
        runtime_mappings_opened=0, control_messages_sent=0, ui_actions_sent=0, arms=[])
    try:
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("Owned GPU qualifier requires CUDA-compatible device")
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
            if len(arrays["decision_monotonic_seconds"]) != 739:
                raise ValueError("Whole original reference must retain 739 observations")
            times = arrays["decision_monotonic_seconds"][:65].copy()
            retinas = arrays["retina_intensity"][:65].copy()
            expected = np.concatenate([arrays["additional_activity"][:65],
                                       arrays["motion_activity"][:65]], axis=2)
        with load_array(vision / "retina-registration.npz", allow_pickle=False) as arrays:
            indices = np.concatenate([arrays["additional_indices"], arrays["motion_indices"]])
            if arrays["view_order"].tolist() != ["front", "left", "right", "rear"]:
                raise ValueError("Four original independent camera identities required")
        gaps = np.diff(times)
        if len(gaps) != 64 or not np.all((gaps > 0) & (gaps <= .08)):
            raise ValueError("Original positive intervals <=80ms required")
        write_json(run / "original-prefix-reference.json", dict(
            episode=entry["id"], role=entry["role"], complete_source_observations=739,
            declared_prefix_observations=65, original_clocks=times.tolist(),
            intervals=gaps.tolist(), original_elapsed_seconds=float(times[-1] - times[0]),
            retina_prefix_sha256=hashlib.sha256(retinas.tobytes()).hexdigest(),
            reference_selected_prefix_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            selected_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            source_response_sha256=entry["responses_sha256"], prefix_chosen_before_execution=True))
        network.eval().requires_grad_(False)
        with torch.no_grad():
            initial = network.steady_state(1., .005, batch_size=4, value=.5)
        initial, _ = detach_exact(initial)
        network.clamp()
        if any(not torch.equal(value.detach().cpu(), weights[k].cpu())
               for k, value in network.state_dict().items() if k in weights):
            raise ValueError("Initialization/clamp changed original author weights")
        initial_hashes = state_digest(initial)
        selected_indices = torch.tensor(indices, dtype=torch.long)
        projection = torch.linspace(.5, 1.5, 4 * len(indices), dtype=torch.float32).reshape(
            4, len(indices), 1)
        initial_selected = initial.nodes.activity[:, selected_indices]
        initial_error = float(abs(initial_selected.cpu().numpy() - expected[0]).max())
        if initial_error > plan["reference_absolute_tolerance"]:
            raise ValueError("Declared initial state differs from original replay")
        initial_scalar = float((initial_selected.square() * projection).mean())
        del initial_selected
        free_names = [name for name, _ in network.named_parameters() if flags[name]]
        snapshots = {}
        recompute_calls = {}
        baseline_numeric_states = []
        static_indices = {}

        def advance_interval(state, params, stimulus, intervals):
            recompute_calls["started"] += 1
            # Backward recomputation runs in an autograd thread whose default
            # device need not inherit the caller. Author allocations stay on
            # the explicit stimulus device; equations and dt are unchanged.
            with torch.device(stimulus.device):
                for dt in intervals:
                    state = network._next_state(params, state, stimulus, dt)
            return state

        for arm in plan["arms"]:
            wall = time.perf_counter()
            network.zero_grad(set_to_none=True)
            for name, p in network.named_parameters():
                p.requires_grad_(flags[name])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            state, _ = detach_exact(initial)
            logical_ticks, decisions, windows, detach_count = 0, [], [], 0
            recompute_calls = {"started": 0}
            for window_start in range(0, 64, 8):
                network.zero_grad(set_to_none=True)
                objective_value = 0.
                for start in range(window_start, window_start + 8, arm["detach_intervals"]):
                    params = network._param_api()
                    loss = torch.zeros((), dtype=torch.float32)
                    for i in range(start, start + arm["detach_intervals"]):
                        network.stimulus.zero(4, 1)
                        network.stimulus.add_input(torch.tensor(retinas[i])[:, None, None, :])
                        stimulus = network.stimulus()[:, 0].clone()
                        intervals = tuple(integration_intervals(float(gaps[i])))
                        if arm["recompute"]:
                            state = checkpoint(advance_interval, state, params, stimulus,
                                intervals, use_reentrant=False, preserve_rng_state=True)
                        else:
                            state = advance_interval(state, params, stimulus, intervals)
                        logical_ticks += len(intervals)
                        selected = state.nodes.activity[:, selected_indices]
                        error = float(abs(selected.detach().cpu().numpy() - expected[i+1]).max())
                        if error > plan["reference_absolute_tolerance"]:
                            raise ValueError("Author trace differs from original neuronal replay")
                        current_numeric = {}
                        aliases = {}
                        for key, value in state_tensors(state).items():
                            if key.endswith(".indices"):
                                cpu = value.detach().cpu().numpy()
                                if key not in static_indices:
                                    static_indices[key] = cpu.copy()
                                elif not np.array_equal(cpu, static_indices[key]):
                                    raise ValueError("Original state reference indices changed")
                                current_numeric[key] = static_indices[key]
                            else:
                                pointer = value.data_ptr()
                                if pointer not in aliases:
                                    aliases[pointer] = value.detach().cpu().numpy().copy()
                                current_numeric[key] = aliases[pointer]
                        if arm["name"] == "detach2_ordinary":
                            baseline_numeric_states.append(current_numeric)
                        baseline_state = baseline_numeric_states[i]
                        state_errors = {key: float(abs(value-baseline_state[key]).max())
                                        for key, value in current_numeric.items()}
                        output_error = float(abs(selected.detach().cpu().numpy()
                            - baseline_state["state.nodes.activity"][:, indices]).max())
                        if plan.get("revision") == 4 and (
                                max(state_errors.values()) > plan["cross_arm_absolute_tolerance"]
                                or output_error > plan["cross_arm_absolute_tolerance"]):
                            raise ValueError("Cross-arm numerical tolerance exceeded")
                        scalar = (selected.square() * projection).mean()
                        if not scalar.requires_grad or not torch.isfinite(scalar):
                            raise ValueError("Finite differentiable local diagnostic required")
                        scalar_value = float(scalar.detach())
                        objective_value += scalar_value / 8
                        loss = loss + scalar / 8
                        decisions.append(dict(index=i+1, held_retina_index=i,
                            original_clock=float(times[i+1]), elapsed=float(gaps[i]),
                            internal_steps=len(intervals), diagnostic_scalar=scalar_value,
                            source_neural_max_error=error, state_hashes=state_digest(state),
                            baseline_full_state_max_errors=state_errors,
                            baseline_selected_activity_max_error=output_error,
                            selected_activity_sha256=tensor_digest(selected),
                            view_channel_means=selected.detach().mean(-1).cpu().tolist()))
                    loss.backward()
                    state, leaves = detach_exact(state)
                    detach_count += 1
                    del loss, selected, scalar, params, stimulus
                gradients = {}
                for name, parameter in network.named_parameters():
                    if not flags[name] and parameter.grad is not None:
                        raise ValueError("Originally fixed author value received gradient")
                    if flags[name]:
                        gradient = parameter.grad
                        if gradient is None or not torch.isfinite(gradient).all():
                            raise ValueError("Missing or nonfinite author-free gradient")
                        gradients[name] = gradient.detach().cpu().numpy().reshape(-1).tolist()
                flattened = np.concatenate([gradients[name] for name in free_names])
                if not np.count_nonzero(flattened):
                    raise ValueError("All author-free gradients are zero")
                windows.append(dict(start=window_start, end=window_start+8,
                    accumulated_diagnostic_objective=objective_value,
                    gradient_elements=len(flattened), nonzero_gradients=int(
                        np.count_nonzero(flattened)), gradients=gradients,
                    state_hashes_after_exact_detach=state_digest(state),
                    detached_state_tensors=leaves, optimizer_updates=0))
                write_json(run / "progress.json", dict(arm=arm["name"],
                    original_intervals_covered=window_start+8, logical_ticks=logical_ticks,
                    optimizer_updates=0, allocated_gpu_bytes=torch.cuda.memory_allocated(),
                    peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated()))
            if state_digest(initial) != initial_hashes:
                raise ValueError("An arm mutated shared declared initial state")
            changed = [k for k, value in network.state_dict().items()
                       if k in weights and not torch.equal(value.detach().cpu(), weights[k].cpu())]
            if changed:
                raise ValueError("Qualifier changed author parameter values")
            peak = torch.cuda.max_memory_allocated()
            if peak > plan["maximum_peak_gpu_tensor_bytes"]:
                raise ValueError("Declared qualifier GPU memory ceiling exceeded")
            report = dict(**arm, windows=windows, decisions=decisions,
                original_intervals=64, prefix_observations=65, logical_internal_ticks=logical_ticks,
                started_interval_evaluations=recompute_calls["started"],
                exact_detach_boundaries=detach_count, recurrent_numeric_resets=0,
                initial_state_hashes=initial_hashes, initial_diagnostic_scalar=initial_scalar,
                initial_neural_max_error=initial_error, initial_observation_retained=True,
                author_parameter_changes=changed, wall_seconds=time.perf_counter()-wall,
                peak_gpu_tensor_bytes=peak,
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
            write_json(run / (arm["name"] + ".json"), report)
            snapshots[arm["name"]] = report
            result["arms"].append({k: v for k, v in report.items()
                                   if k not in ("windows", "decisions")})
            network.zero_grad(set_to_none=True)
            del state, windows, decisions, gradients, flattened
            gc.collect()
            torch.cuda.empty_cache()
        baseline = snapshots["detach2_ordinary"]
        cross_arm_hashes_exact = True
        for arm in snapshots.values():
            for old, new in zip(baseline["decisions"], arm["decisions"], strict=True):
                exact = (old["state_hashes"] == new["state_hashes"]
                    and old["selected_activity_sha256"] == new["selected_activity_sha256"]
                    and old["diagnostic_scalar"] == new["diagnostic_scalar"])
                cross_arm_hashes_exact = cross_arm_hashes_exact and exact
                if plan.get("revision") == 4:
                    if abs(old["diagnostic_scalar"]-new["diagnostic_scalar"]) > (
                            plan["cross_arm_absolute_tolerance"]):
                        raise ValueError("Cross-arm diagnostic tolerance exceeded")
                elif not exact:
                    raise ValueError("Detach or recomputation changed numerical state/output")
        reference_gradient_errors, credit_gradient_changes = [], []
        has_recompute_control = "detach2_recompute" in snapshots
        repeat_name = "detach2_repeat" if "detach2_repeat" in snapshots else "detach2_recompute"
        has_repeat_control = repeat_name in snapshots
        repeat_windows = (snapshots[repeat_name]["windows"] if has_repeat_control
                          else baseline["windows"])
        long_name = "detach8_recompute" if has_recompute_control else "detach8_ordinary"
        for base, recomputed, long in zip(baseline["windows"], repeat_windows,
                snapshots[long_name]["windows"], strict=True):
            original = np.concatenate([base["gradients"][name] for name in free_names])
            repeat = np.concatenate([recomputed["gradients"][name] for name in free_names])
            longer = np.concatenate([long["gradients"][name] for name in free_names])
            np.testing.assert_allclose(original, repeat, rtol=plan["gradient_relative_tolerance"],
                                       atol=plan["gradient_absolute_tolerance"])
            reference_gradient_errors.append(float(abs(original-repeat).max()))
            repeat_noise = float(abs(original-repeat).max())
            credit_change = float(abs(original-longer).max())
            if plan.get("revision") == 4 and credit_change <= 10 * (repeat_noise + 1e-12):
                raise ValueError("Temporal gradient difference does not exceed repeat noise")
            credit_gradient_changes.append(dict(start=base["start"], end=base["end"],
                maximum_absolute_gradient_change=credit_change,
                repeated_short_gradient_max_difference=repeat_noise,
                short_norm=float(np.linalg.norm(original)),
                long_norm=float(np.linalg.norm(longer))))
        result.update(status="completed_author_temporal_window_qualifier",
            numerical_state_output_hashes_exact_across_arms=cross_arm_hashes_exact,
            within_arm_numerical_detach_exact=True,
            cross_arm_numerical_agreement_within_declared_tolerance=True,
            repeated_short_gradient_max_error=(
                max(reference_gradient_errors) if has_repeat_control else None),
            ordinary_vs_recompute_gradient_max_error=(
                max(reference_gradient_errors) if has_recompute_control else None),
            activation_recomputation_qualified=has_recompute_control,
            temporal_credit_gradient_changes=credit_gradient_changes,
            original_elapsed_seconds=float(times[-1]-times[0]),
            original_clocks_held_retinas_preserved=True, original_parameter_values_unchanged=True,
            original_trainable_elements=sum(p.numel() for name, p in network.named_parameters()
                                            if flags[name]),
            fixed_parameter_elements=sum(p.numel() for name, p in network.named_parameters()
                                         if not flags[name]),
            scope=("Declared TRAIN prefix8-window state/gradient/memory preflight; "
                   "no fitting or predictive qualification"))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run / "result.json", result)


if __name__ == "__main__":
    main()



