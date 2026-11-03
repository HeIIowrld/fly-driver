"""Whole-TRAIN-history temporal credit experiment, original author and fold-only readouts."""

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

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.turn_residual import current_turn_prior
from scripts.prepare_flyvis_preview_folds import TRAIN_IDS, fingerprint, registered_cpu, score
from scripts.probe_flyvis_registered_gradient import registered_features
from scripts.probe_flyvis_temporal_window import detach_exact, state_digest
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_visual_body_preview import body_features, direct_body_reference


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def parameter_digest(network):
    digest = hashlib.sha256()
    for name, parameter in network.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def temporal_window_layout(total_intervals, detach_intervals):
    """Equal optimizer cadence with exact original tail and no skipped intervals."""
    if (type(total_intervals) is not int or total_intervals < 1
            or type(detach_intervals) is not int or detach_intervals not in (2, 8)):
        raise ValueError("Positive original interval count and detach2/8 required")
    for start in range(0, total_intervals, 8):
        end = min(start+8, total_intervals)
        yield start, end, tuple((substart, min(substart+detach_intervals, end))
                               for substart in range(start, end, detach_intervals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    expected_arms = [dict(name="detach2", detach_intervals=2),
                     dict(name="detach8", detach_intervals=8)]
    if (plan["schema"] != "flybrain.train_only_author_temporal_credit.v1"
            or plan["train_ids"] != TRAIN_IDS or plan["arms"] != expected_arms
            or plan["optimizer_window_intervals"] != 8 or plan["epochs"] != 1
            or plan["learning_rate"] != 1e-5 or plan["relative_parameter_regularization"] != .001
            or plan["gradient_norm_limit"] != 1. or plan["readout_trainable"] is not False
            or plan["known_development_arrays_allowed"] is not False
            or plan["expected_author_optimizer_updates"] != 1116
            or plan["candidate_rule"] != "final weights after both complete fit histories only"):
        raise ValueError("Fixed matched whole-history TRAIN temporal experiment required")
    for path, digest in plan["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Sealed temporal experiment input changed: " + path)
    qualifier = read(Path(plan["full_graph_qualifier"]) / "result.json")
    if qualifier["status"] != "completed_full_preview_window_qualifier":
        raise ValueError("Actual every-observation preview8 graph qualifier required")
    folds_root = Path(plan["fold_setup"])
    fold_proof = read(folds_root / "result.json")
    if (fold_proof["status"] != "completed_cpu_train_history_fold_setup"
            or not fold_proof["all_fold_normalizations_and_readouts_fit_only_two"]
            or fold_proof["known_development_arrays_opened"] != 0):
        raise ValueError("Completed fold-only readouts/normalization required")
    parity = read(Path(plan["fold_gpu_parity"]) / "result.json")
    if (parity["status"] != "passed_fold_cpu_gpu_feature_readout_parity"
            or parity["cases"] != 36 or len(parity["records"]) != 36
            or not parity["all_pose_support_exact"]
            or not parity["all_gpu_routes_use_fold_calibration"]
            or parity["known_development_arrays_opened"] != 0
            or parity["original_fold_setup_result_sha256"] != sha256(folds_root / "result.json")):
        raise ValueError("Actual sealed fold CPU/GPU feature/readout parity required")
    source = read(Path(plan["registered_data"]) / "dataset.json")
    sensor = read(Path(plan["sensor_replay"]) / "result.json")
    entries = [e for e in source["episodes"] if e["role"] == "train"]
    if [e["id"] for e in entries] != TRAIN_IDS:
        raise ValueError("Only original TRAIN3 may be opened")
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), original_external_roles_unchanged=True,
        known_development_arrays_opened=0, original_payloads_copied=0,
        final_candidates_only=True, original_author_fresh_per_fold_arm=True))
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = dict(status="failed", known_development_arrays_opened=0,
        evaluation_arrays_opened=0, original_payloads_copied=0, actor_deployed=False,
        independent_final_validation=False, physiological_function_verified=False,
        goal_achieved=False, author_optimizer_updates=0, readout_optimizer_updates=0,
        control_messages_sent=0, ui_actions_sent=0, runtime_mappings_opened=0, records=[])
    updates, records = 0, []
    try:
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("Owned GPU temporal training required")
        os.environ["FLYVIS_ROOT_DIR"] = str(Path(plan["cache_root"]).resolve())
        os.environ["MPLBACKEND"] = "Agg"
        install_datamate_cache_compat(plan["cache_root"])
        import flyvis

        torch.set_default_device("cuda")
        episodes = []
        for entry in entries:
            original = next(e for e in sensor["episodes"] if e["id"] == entry["id"])
            vision = Path(original["output"])
            if sha256(vision / "responses.npz") != original["responses_sha256"]:
                raise ValueError("Original TRAIN sensor history changed")
            with load_array(vision / "responses.npz", allow_pickle=False) as arrays:
                clock = arrays["decision_monotonic_seconds"].copy()
                retina = arrays["retina_intensity"].copy()
                activity = np.concatenate([arrays["additional_activity"],
                                           arrays["motion_activity"]], axis=2)
            with load_array(entry["targets_file"], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(clock, arrays["observation_time"])
                target = arrays["preview"].reshape(-1, 6).copy()
                valid = np.repeat(arrays["preview_valid"], 2, 1)
            body, mask, _ = direct_body_reference(entry, clock, sha256(plan["body_contract"]))
            with load_array(vision / "retina-registration.npz", allow_pickle=False) as arrays:
                indices = np.concatenate([arrays["additional_indices"], arrays["motion_indices"]])
                rays, origins = arrays["body_directions"].copy(), arrays["body_origins_m"].copy()
            if (len(clock) != 739 or not np.all((np.diff(clock) > 0)
                                               & (np.diff(clock) <= .08))):
                raise ValueError("Whole739-observation original TRAIN clocks required")
            episodes.append(dict(entry, clock=clock, retina=retina, activity=activity,
                y=target, valid=valid, body=body, body_valid=mask, indices=indices,
                rays=rays, origins=origins))
        model_dir = Path(plan["model_dir"])
        config = yaml.safe_load((model_dir / "_meta.yaml").read_text(encoding="utf8"))
        weights = torch.load(model_dir / "best_chkpt", map_location="cpu",
                             weights_only=True)["network"]
        for fold_index, heldout in enumerate(TRAIN_IDS):
            fold_folder = folds_root / heldout
            fold = read(fold_folder / "result.json")
            fit_ids = [key for key in TRAIN_IDS if key != heldout]
            if (fold["fit_history_ids"] != fit_ids or fold["readout_fit_ids"] != fit_ids
                    or fold["all_moments_fit_ids"] != fit_ids
                    or not fold["holdout_perturbation_moments_exact"]):
                raise ValueError("Fold holdout leaked into fitting")
            saved = torch.load(fold_folder / "vision_body.pt", map_location="cpu",
                               weights_only=False)
            if (saved["fitted_history_ids"] != fit_ids or saved["heldout_history"] != heldout
                    or saved["original_author_checkpoint_sha256"] != (
                        plan["original_author_checkpoint_sha256"])):
                raise ValueError("Exact fold-only original-author readout required")
            with load_array(fold_folder / "normalization.npz", allow_pickle=False) as arrays:
                normalization = {key: arrays[key].copy() for key in arrays.files}
            for key, value in normalization.items():
                np.testing.assert_array_equal(value, saved[key])
            count = np.concatenate([e["valid"] for e in episodes if e["id"] in fit_ids]).sum(0)
            np.testing.assert_array_equal(count, normalization["target_valid_counts"])
            for e in episodes:
                for key in ("clock", "valid", "body", "body_valid"):
                    if fingerprint(e[key]) != fold["source_array_fingerprints"][e["id"]][key]:
                        raise ValueError("Original clocks/masks/body differ from fold setup")
            with torch.device("cpu"):
                original_features = {e["id"]: registered_cpu(e,
                    normalization["channel_mean"], normalization["channel_scale"],
                    batch_size=plan["pool_batch_size"]) for e in episodes}
            for arm in plan["arms"]:
                folder = run / heldout / arm["name"]
                folder.mkdir(parents=True)
                torch.manual_seed(plan["seed"] + fold_index)
                network = flyvis.Network(**config["config"]["network"]).to("cuda")
                flags = {name: p.requires_grad for name, p in network.named_parameters()}
                network.load_state_dict(weights, strict=True)
                initial_parameter_hash = parameter_digest(network)
                implementation = Path(inspect.getsourcefile(type(network)))
                write_json(folder / "author-implementation.json", dict(path=str(implementation),
                    sha256=sha256(implementation), original_trainable_flags=flags))
                readout = RegisteredPreview().to("cuda").eval().requires_grad_(False)
                readout.load_state_dict(saved["model"])
                optimizer = torch.optim.Adam([p for name, p in network.named_parameters()
                                              if flags[name]], lr=plan["learning_rate"])
                originals = {name: p.detach().clone() for name, p in network.named_parameters()
                             if flags[name]}
                parameter_scales = {name: p.abs().mean().clamp_min(.01)
                                    for name, p in originals.items()}
                feature_mean = torch.tensor(normalization["feature_mean"], dtype=torch.float32)
                feature_scale = torch.tensor(normalization["feature_scale"], dtype=torch.float32)
                target_mean = torch.tensor(normalization["target_mean"], dtype=torch.float32)
                target_scale = torch.tensor(normalization["target_scale"], dtype=torch.float32)
                loss_weight = torch.tensor(1. / (6 * count), dtype=torch.float32)
                arm_updates, fit_records = 0, []
                torch.cuda.reset_peak_memory_stats()

                def setup(e, differentiable, *, network=network, flags=flags,
                          normalization=normalization, saved=saved):
                    network.eval().requires_grad_(False)
                    with torch.no_grad():
                        initial = network.steady_state(1., .005, batch_size=4, value=.5)
                    initial, _ = detach_exact(initial)
                    network.clamp()
                    if differentiable:
                        for name, p in network.named_parameters():
                            p.requires_grad_(flags[name])
                    routed = body_features(e["body"], e["body_valid"],
                        normalization["body_mean"], normalization["body_scale"],
                        saved["body_route_indices"])
                    return initial, torch.tensor(e["indices"], dtype=torch.long), torch.tensor(
                        routed)

                def advance(e, i, state, params, *, network=network):
                    network.stimulus.zero(4, 1)
                    network.stimulus.add_input(torch.tensor(e["retina"][i])[:, None, None, :])
                    stimulus = network.stimulus()[:, 0].clone()
                    intervals = integration_intervals(float(e["clock"][i+1]-e["clock"][i]))
                    for dt in intervals:
                        state = network._next_state(params, state, stimulus, dt)
                    return state, len(intervals)

                def features(e, i, state, indices, *, normalization=normalization):
                    return registered_features(state.nodes.activity[:, indices], e["rays"],
                        e["origins"], e["body"][i:i+1], e["body_valid"][i:i+1],
                        normalization["channel_mean"], normalization["channel_scale"])

                def predict(x, b, *, readout=readout, feature_mean=feature_mean,
                            feature_scale=feature_scale):
                    return readout(((x-feature_mean)/feature_scale).clamp(-5, 5), b)

                for e in (e for e in episodes if e["id"] in fit_ids):
                    wall = time.perf_counter()
                    state, indices, routed = setup(e, True)
                    with torch.no_grad():
                        initial_feature = features(e, 0, state, indices)
                        initial_prediction = predict(initial_feature, routed[:1])
                        initial_target = torch.tensor(e["y"][:1], dtype=torch.float32)
                        initial_loss = ((initial_prediction-(initial_target-target_mean)/(
                            target_scale)).square()*torch.tensor(e["valid"][:1])*(
                                loss_weight)).sum()
                    if arm_updates == 0:
                        error = float(abs(initial_feature.cpu().numpy()[0]-(
                            original_features[e["id"]]["intact"][0])).max())
                        if error > plan["initial_feature_absolute_tolerance"]:
                            raise ValueError("Fold-normalized original feature parity failed")
                    windows, gradient_counts, ticks, detach_count = [], np.zeros(6, int), 0, 0
                    for start, end, subchunks in temporal_window_layout(
                            len(e["clock"])-1, arm["detach_intervals"]):
                        optimizer.zero_grad(set_to_none=True)
                        before_hash = parameter_digest(network)
                        data_value, detach_records = 0., []
                        for substart, subend in subchunks:
                            params = network._param_api()
                            data_loss = torch.zeros((), dtype=torch.float32)
                            for i in range(substart, subend):
                                state, internal = advance(e, i, state, params)
                                ticks += internal
                                x = features(e, i+1, state, indices)
                                if arm_updates == 0:
                                    error = float(abs(x.detach().cpu().numpy()[0]-(
                                        original_features[e["id"]]["intact"][i+1])).max())
                                    if error > plan["initial_feature_absolute_tolerance"]:
                                        raise ValueError("Unupdated fold feature parity failed")
                                output = predict(x, routed[i+1:i+2])
                                target = torch.tensor(e["y"][i+1:i+2], dtype=torch.float32)
                                available = torch.tensor(e["valid"][i+1:i+2])
                                data_loss = data_loss+((output-(target-target_mean)/(
                                    target_scale)).square()*available*loss_weight).sum()
                                gradient_counts += e["valid"][i+1].astype(int)
                            if not torch.isfinite(data_loss):
                                raise ValueError("Nonfinite temporal data loss")
                            data_value += float(data_loss.detach())
                            data_loss.backward()
                            state, leaves = detach_exact(state)
                            detach_count += 1
                            detach_records.append(dict(index=subend, exact=True,
                                state_tensors=leaves, state_hashes=state_digest(state)))
                            if parameter_digest(network) != before_hash:
                                raise ValueError("Parameter changed inside optimizer window")
                            del data_loss, params, x, output, target, available
                        regularizer = torch.stack([((p-originals[name])/(
                            parameter_scales[name])).square().mean()
                            for name, p in network.named_parameters() if flags[name]]).mean()
                        penalty = plan["relative_parameter_regularization"]*regularizer
                        if not torch.isfinite(penalty):
                            raise ValueError("Nonfinite once-per-window parameter penalty")
                        penalty.backward()
                        nonzero = 0
                        for name, p in network.named_parameters():
                            if not flags[name] and p.grad is not None:
                                raise ValueError("Fixed author value received a gradient")
                            if p.grad is not None:
                                if not torch.isfinite(p.grad).all():
                                    raise ValueError("Nonfinite author gradient")
                                nonzero += int(torch.count_nonzero(p.grad))
                        norm = torch.nn.utils.clip_grad_norm_(
                            [p for name, p in network.named_parameters() if flags[name]],
                            plan["gradient_norm_limit"], error_if_nonfinite=True)
                        if any(p.grad is not None for p in readout.parameters()):
                            raise ValueError("Frozen fold readout received a gradient")
                        if torch.cuda.max_memory_allocated() > plan["maximum_peak_gpu_bytes"]:
                            raise ValueError("Full temporal experiment memory exceeded14GB")
                        optimizer.step()
                        network.clamp()
                        updates += 1
                        arm_updates += 1
                        windows.append(dict(start=start, end=end, original_interval_count=end-start,
                            data_loss=data_value, regularizer=float(regularizer.detach()),
                            penalty=float(penalty.detach()), regularizer_evaluations=1,
                            gradient_norm_before_clip=float(norm), nonzero_gradients=nonzero,
                            parameter_hash_before=before_hash,
                            parameters_constant_inside_window=True,
                            parameter_hash_after=parameter_digest(network),
                            detach_boundaries=detach_records, author_optimizer_updates=arm_updates))
                        if updates % 10 == 0:
                            write_json(run / "progress.json", dict(heldout=heldout, arm=arm,
                                episode=e["id"], original_decision=end,
                                author_optimizer_updates=updates,
                                arm_optimizer_updates=arm_updates,
                                data_loss=data_value, gradient_norm_before_clip=float(norm),
                                peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated()))
                        del regularizer, penalty
                    np.testing.assert_array_equal(gradient_counts, e["valid"][1:].sum(0))
                    record = dict(episode=e["id"], original_role="train", observations=739,
                        original_intervals=738, original_elapsed_seconds=float(
                            e["clock"][-1]-e["clock"][0]), logical_internal_ticks=ticks,
                        original_clock_fingerprint=fingerprint(e["clock"]),
                        original_mask_fingerprint=fingerprint(e["valid"]),
                        gradient_target_counts=gradient_counts.tolist(),
                        initial_gray_detached_target_counts=e["valid"][0].astype(int).tolist(),
                        initial_gray_loss=float(initial_loss),
                        initial_gray_state_retained=True, recurrent_numeric_resets=0,
                        detach_boundaries=detach_count, windows=windows,
                        wall_seconds=time.perf_counter()-wall)
                    write_json(folder / (e["id"]+"-training.json"), record)
                    fit_records.append({k: v for k, v in record.items() if k != "windows"})
                changed = [k for k, v in network.state_dict().items()
                           if k in weights and not torch.equal(v.detach().cpu(), weights[k].cpu())]
                if not changed or any(not flags.get(k, False) for k in changed):
                    raise ValueError("No author adaptation or fixed author value changed")
                readout_changed = [k for k, v in readout.state_dict().items()
                                   if not torch.equal(v.detach().cpu(), saved["model"][k].cpu())]
                if readout_changed:
                    raise ValueError("Frozen fold readout values changed")
                changed_elements = {name: int(torch.count_nonzero(p.detach().cpu()-(
                    weights[name].cpu()))) for name, p in network.named_parameters()
                    if flags[name]}
                network.eval().requires_grad_(False)
                torch.save(dict(network=network.state_dict(), plan=plan, heldout=heldout,
                    fit_history_ids=fit_ids, arm=arm, original_trainable_flags=flags,
                    author_optimizer_updates=arm_updates, readout_trainable=False,
                    actor_policy=False), folder / "adapted-author.pt")
                holdout = next(e for e in episodes if e["id"] == heldout)
                state, indices, routed = setup(holdout, False)
                learned_activity = []
                with torch.no_grad():
                    params = network._param_api()
                    for i in range(len(holdout["clock"])):
                        if i:
                            state, _ = advance(holdout, i-1, state, params)
                        learned_activity.append(state.nodes.activity[:, indices].cpu().numpy())
                        if i % 100 == 0:
                            write_json(run / "progress.json", dict(heldout=heldout, arm=arm,
                                holdout_decision=i, frozen_final_weights=True,
                                author_optimizer_updates=updates))
                learned_activity = np.asarray(learned_activity)
                with torch.device("cpu"):
                    held_features = registered_cpu(dict(holdout, activity=learned_activity),
                        normalization["channel_mean"], normalization["channel_scale"],
                        batch_size=plan["pool_batch_size"])
                np.testing.assert_array_equal(held_features["intact"][:, 896:],
                                               original_features[heldout]["intact"][:, 896:])
                normalized = {key: ((v-normalization["feature_mean"])/(
                    normalization["feature_scale"])).clip(-5, 5).astype(np.float32)
                    for key, v in held_features.items()}
                variants = dict(normalized)
                replacement = normalized["intact"].copy()
                replacement[:, :896] = 0
                variants["neural_feature_mean"] = replacement
                replacement = held_features["intact"].copy()
                replacement[:, :896] = 0
                variants["retinal_neural_mean"] = ((replacement-normalization["feature_mean"])/(
                    normalization["feature_scale"])).clip(-5, 5).astype(np.float32)
                predictions = {}
                with torch.no_grad():
                    for name, value in variants.items():
                        inferred = readout(torch.tensor(value), routed).cpu().numpy()
                        predictions["adapted__"+name] = inferred*normalization["target_scale"]+(
                            normalization["target_mean"])
                    body_saved = torch.load(fold_folder / "body_only.pt", map_location="cpu",
                                            weights_only=False)
                    body_model = RegisteredPreview().to("cuda").eval().requires_grad_(False)
                    body_model.load_state_dict(body_saved["model"])
                    inferred = body_model(torch.zeros((739, 1024)), routed).cpu().numpy()
                    predictions["body_only"] = inferred*normalization["target_scale"]+(
                        normalization["target_mean"])
                predictions["current_turn_prior"], _ = current_turn_prior(
                    holdout["body"], holdout["body_valid"])
                predictions["fit_target_mean"] = np.broadcast_to(
                    normalization["target_mean"], holdout["y"].shape)
                predictions["zero"] = np.zeros_like(holdout["y"])
                metrics = score(holdout, predictions, normalization["target_scale"])
                np.savez_compressed(folder / "holdout-predictions.npz", **predictions,
                                    observation_time=holdout["clock"])
                arm_record = dict(heldout_history=heldout, heldout_external_role="train",
                    fit_history_ids=fit_ids, arm=arm, fit_records=fit_records,
                    author_optimizer_updates=arm_updates, readout_optimizer_updates=0,
                    initial_author_parameter_hash=initial_parameter_hash,
                    final_author_parameter_hash=parameter_digest(network),
                    author_changed_parameter_names=changed,
                    author_changed_elements=changed_elements,
                    original_fixed_author_values_unchanged=True,
                    frozen_fold_readout_values_unchanged=True,
                    peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated(),
                    holdout_observations=739, holdout_intervals=738,
                    holdout_clock_fingerprint=fingerprint(holdout["clock"]),
                    holdout_mask_fingerprint=fingerprint(holdout["valid"]),
                    holdout_recurrent_resets=0, holdout_frozen_final_weights=True,
                    all_pose_support_masks_exact=True,
                    holdout_metrics=metrics, original_fold_metrics=fold["holdout_metrics"],
                    final_checkpoint_sha256=sha256(folder / "adapted-author.pt"),
                    predictions_sha256=sha256(folder / "holdout-predictions.npz"),
                    candidate_rule=plan["candidate_rule"], actor_deployed=False)
                write_json(folder / "result.json", arm_record)
                records.append(arm_record)
                result["records"] = records
                result["author_optimizer_updates"] = updates
                write_json(run / "partial-result.json", result)
                del network, optimizer, readout, body_model, state, params, originals
                del setup, advance, features, predict
                del learned_activity, held_features, normalized, variants, predictions
                gc.collect()
                torch.cuda.empty_cache()
        if updates != plan["expected_author_optimizer_updates"] or len(records) != 6:
            raise ValueError("Incomplete declared six-arm temporal experiment")
        paired = []
        for heldout in TRAIN_IDS:
            short = next(r for r in records if r["heldout_history"] == heldout
                         and r["arm"]["name"] == "detach2")
            long = next(r for r in records if r["heldout_history"] == heldout
                        and r["arm"]["name"] == "detach8")
            if short["initial_author_parameter_hash"] != long["initial_author_parameter_hash"]:
                raise ValueError("Temporal arms did not start from exact same original author")
            paired.append(dict(heldout=heldout,
                detach2_standardized_mse=short["holdout_metrics"]["adapted__intact"][
                    "standardized_mse"],
                detach8_standardized_mse=long["holdout_metrics"]["adapted__intact"][
                    "standardized_mse"],
                original_standardized_mse=short["original_fold_metrics"]["vision_body__intact"][
                    "standardized_mse"],
                body_only_standardized_mse=short["holdout_metrics"]["body_only"][
                    "standardized_mse"]))
        result.update(status="completed_train_only_author_temporal_credit",
            records=records, author_optimizer_updates=updates, paired_fold_comparisons=paired,
            all_final_candidates_retained=True, fixed_author_values_unchanged=True,
            all_readouts_normalizations_fold_only_and_frozen=True,
            original_full_clock_mask_body_values_unchanged=True,
            new_unique_original_observations=0, original_unique_train_observations=2217,
            repeated_fit_observation_exposures=12*739,
            motor_transfer_qualified=False, visual_anticipation_final_qualified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error),
                      author_optimizer_updates=updates, records=records)
        raise
    finally:
        write_json(run / "result.json", result)


if __name__ == "__main__":
    main()
