"""Train view-local transduction on sealed teacher episodes with persistent recurrent state."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import (
    calibrate_photometry,
    condition_independent_pedals,
    condition_native_steering,
    configure_trainable_groups,
    control_label_weights,
    episode_batches,
    episode_chunk,
    load_episodes,
    replay_chunk,
    replay_metrics,
    replay_output_activity,
)
from flybrain_drive.policy_anchor import (
    anchored_selection_score,
    policy_anchor_loss,
    prepare_policy_anchors,
    replay_anchor_mse,
)
from flybrain_drive.vehicle_circuit import encode_positive_gain, positive_gain
from flybrain_drive.visual_age_calibration import calibrate_visual_age_decay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiview-steering-pilot.json")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    device = torch.device(config.get('device', 'cpu'))
    if device.type not in ('cpu', 'cuda') or (
            device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested CPU or CUDA/ROCm training device must be available')
    update_scope = config.get('optimizer_update_scope', 'chunk')
    if update_scope not in ('chunk', 'epoch'):
        raise ValueError('Explicit chunk or epoch optimizer update scope required')
    if (
        type(config["epochs"]) is not int
        or config["epochs"] < 1
        or type(config["chunk_length"]) is not int
        or config["chunk_length"] < 1
        or not np.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
    ):
        raise ValueError("Positive finite optimization budgets required")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(config["seed"])
    run = begin_run(
        args.output,
        {
            "configuration": config,
            "teacher_manifest_sha256": sha256(Path(args.teacher) / "dataset.json"),
            "state_contract": "reset per epoch/episode; detach without reset at chunk boundaries",
            "optimizer_contract": "one update per " + update_scope
                                  + "; detach each chunk; recurrent structural weights fixed",
        },
    )
    try:
        if sha256(config["source_checkpoint"]) != config["source_checkpoint_sha256"]:
            raise ValueError("Source checkpoint hash differs")
        source = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=True)
        graph = {
            k: np.asarray(
                v, dtype={"pre": np.int64, "post": np.int64, "weights": np.float32}.get(k)
            )
            for k, v in source["graph"].items()
        }
        constructor = copy.deepcopy(source["constructor"])
        source_gain_mode = constructor.get('readout_gain_parameterization', 'softplus')
        target_gain_mode = config.get('readout_gain_parameterization', source_gain_mode)
        constructor['readout_gain_parameterization'] = target_gain_mode
        # Re-encode only readout scales; preserve the realized source gains and graph.
        source = copy.deepcopy(source)
        if target_gain_mode != source_gain_mode:
            for name in ('steering_log_gain', 'pedal_log_gain'):
                key = 'readout.' + name
                source['model'][key] = encode_positive_gain(
                    positive_gain(source['model'][key], source_gain_mode), target_gain_mode)
        if config['action_profile'] == 'normalized_ac_absolute' and 'pedal_mode' not in config:
            raise ValueError('Native AC training requires explicit pedal_mode')
        if 'pedal_mode' in config:
            constructor['pedal_mode'] = config['pedal_mode']
        if 'organ_validity_mode' in config:
            constructor['organ_validity_mode'] = config['organ_validity_mode']
        if 'activation_gradient' in config:
            constructor['activation_gradient'] = config['activation_gradient']
            constructor['surrogate_negative_slope'] = config['surrogate_negative_slope']
        if "circuit_sparse_backend" in config:
            constructor["sparse_backend"] = config["circuit_sparse_backend"]
        deployment_backend = config.get('deployment_sparse_backend',
                                        constructor.get('sparse_backend', 'coo'))
        if device.type == 'cuda':
            constructor['sparse_backend'] = config.get('training_sparse_backend', 'cached_csr_gpu')
            if constructor['sparse_backend'] == 'cached_csr_cpu':
                raise ValueError('GPU training requires a device-capable COO or cached CSR backend')
            torch.cuda.reset_peak_memory_stats(device)
        model = MultiViewCircuitController(graph, **constructor)
        model.load_state_dict(source["model"])
        model.to(device)
        model.chunk_validated_training = config.get('chunk_validated_training', False)
        if type(model.chunk_validated_training) is not bool:
            raise ValueError('Explicit boolean chunk validation mode required')
        episodes, manifest = load_episodes(args.teacher, model.sensory_calibration)
        for episode in episodes:
            episode['observation'] = {k: v.to(device) for k, v in episode['observation'].items()}
            for key in ('action', 'action_valid'):
                episode[key] = episode[key].to(device)
        if manifest.get('expert_labels_available') is False:
            raise ValueError('Query an expert before training on recorded learner requests')
        if manifest.get('teacher_label_contract') is not None and (
                config.get('teacher_label_contract') != manifest['teacher_label_contract']):
            raise ValueError('Training must explicitly match the sealed teacher task contract')
        if (
            manifest["action_profile"] != config["action_profile"]
            or manifest["brain_ticks_per_decision"] != config["brain_ticks_per_decision"]
            or not np.isclose(manifest["decision_dt_seconds"], config["control_dt_seconds"])
        ):
            raise ValueError("Training action profile or timing differs from teacher dataset")
        chunk, ticks = config["chunk_length"], config["brain_ticks_per_decision"]
        batch_maximum = config.get("episodes_per_batch")
        batches = episode_batches(episodes, batch_maximum)
        brake_weight = config.get('brake_positive_weight', 1.)
        brake_threshold = config.get('brake_label_threshold', .05)
        coast_weight = config.get('throttle_coast_weight', 1.)
        coast_threshold = config.get('throttle_coast_threshold', .005)
        if (brake_weight != 1 or coast_weight != 1) and (
                config['action_profile'] != 'normalized_ac_absolute'):
            raise ValueError('Control label weighting requires native AC action semantics')

        anchors, anchor_coefficient = {}, 0.

        def metrics_for_current_model():
            callback = (lambda prediction, batch, offset:
                        policy_anchor_loss(prediction, batch, offset, anchors)) if anchors else None
            metrics = replay_metrics(model, episodes, chunk, ticks, batch_maximum,
                brake_weight, brake_threshold, chunk_metric=callback,
                throttle_coast_weight=coast_weight, throttle_coast_threshold=coast_threshold)
            if anchors:
                metrics['policy_anchor_mse'] = metrics.pop('extra_chunk_metric')
            return metrics

        def selection_score(metrics):
            return anchored_selection_score(metrics, anchor_coefficient)
        original = None
        normalization = config.get("photometric_normalization")
        if normalization:
            original = metrics_for_current_model()
            constructor["photometric_normalization"] = True
            model = MultiViewCircuitController(graph, **constructor).to(device)
            model.chunk_validated_training = config.get('chunk_validated_training', False)
            missing, unexpected = model.load_state_dict(source["model"], strict=False)
            if (unexpected or set(missing) - {"photometric_mean", "photometric_scale",
                                              "photometric_seen"}):
                raise ValueError("Only explicit new photometric buffers may be absent")
            calibration = calibrate_photometry(model, episodes, **normalization)
            np.savez_compressed(run / "photometry.npz",
                mean=model.photometric_mean.cpu().numpy(),
                scale=model.photometric_scale.cpu().numpy(),
                seen=model.photometric_seen.cpu().numpy())
            calibration["sha256"] = sha256(run / "photometry.npz")
            write_json(run / "photometry.json", calibration)
        before = metrics_for_current_model()
        condition_gain = config.get("condition_steering_gain", True)
        if type(condition_gain) is not bool:
            raise ValueError("Explicit steering gain conditioning flag required")
        if config["action_profile"] == "mujoco_curvature_proxy_only" and condition_gain:
            rms = before["output_rms"][0]
            if not np.isfinite(rms) or rms <= 1e-12:
                raise ValueError("No train-input output signal for gain conditioning")
            old_gain = float(positive_gain(
                model.readout.steering_log_gain, target_gain_mode).detach())
            gain = old_gain * config["target_initial_steering_rms"] / rms
            if not 0 < gain <= config["max_steering_gain"]:
                raise ValueError("Required steering gain exceeds declared calibration budget")
            with torch.no_grad():
                model.readout.steering_log_gain.copy_(encode_positive_gain(
                    model.readout.steering_log_gain.new_tensor(gain), target_gain_mode))
        else:
            gain = None
        reconditioned = config["action_profile"] == "mujoco_curvature_proxy_only" and condition_gain
        pedal_conditioning = config.get('pedal_conditioning')
        native_steering = config.get('native_steering_conditioning')
        activity = None
        if pedal_conditioning or native_steering:
            activity = replay_output_activity(model, episodes, chunk, ticks, batch_maximum)
        if native_steering:
            if manifest['action_profile'] != 'normalized_ac_absolute':
                raise ValueError('Native steering conditioning requires native action profile')
            steering_report = condition_native_steering(model.readout, activity, **native_steering)
            write_json(run / 'native-steering-conditioning.json', steering_report)
            gain, reconditioned = steering_report['gain_after'], True
        if pedal_conditioning:
            if manifest['action_profile'] != 'normalized_ac_absolute':
                raise ValueError('Pedal conditioning for training requires native AC action labels')
            report = condition_independent_pedals(model.readout, activity, **pedal_conditioning)
            write_json(run / 'pedal-conditioning.json', report)
        conditioned = metrics_for_current_model() if reconditioned or pedal_conditioning else before
        write_json(
            run / "conditioning.json",
            {
                "before": before,
                "original_before_photometric_calibration": original,
                "conditioned": conditioned,
                "steering_gain": gain,
                "teacher_labels_used_for_gain": False,
                "gain_source": "training sensory output RMS" if reconditioned
                               else "inherited source checkpoint gain, unchanged",
                "gain_reconditioned": reconditioned,
            },
        )
        if (manifest['action_profile'] == 'normalized_ac_absolute'
                and 'trainable_groups' not in config):
            raise ValueError('Native AC training requires explicit trainable_groups')
        trainable_groups = config.get('trainable_groups', ['visual'])
        parameters = configure_trainable_groups(model, trainable_groups,
                                                config.get('organ_group_subset'))
        trainable_parameters = {name: p.numel() for name, p in model.named_parameters()
                                if p.requires_grad}
        write_json(run / 'trainable.json', dict(groups=trainable_groups,
            parameters=trainable_parameters, organ_group_subset=config.get('organ_group_subset'),
            recurrent_weights_trainable=False))
        gain_lr = config.get('readout_gain_learning_rate')
        if gain_lr is not None:
            if (target_gain_mode != 'log' or not np.isfinite(gain_lr) or not 0 < gain_lr <= 1):
                raise ValueError('Relative gain learning rate requires log gains and (0, 1] rate')
            gain_parameters = [p for p in (model.readout.steering_log_gain,
                                           model.readout.pedal_log_gain) if p.requires_grad]
            gain_ids = {id(p) for p in gain_parameters}
            optimizer_parameters = [dict(params=[p for p in parameters if id(p) not in gain_ids])]
            if gain_parameters:
                optimizer_parameters.append(dict(params=gain_parameters, lr=gain_lr))
        else:
            optimizer_parameters = parameters
        gain_before_training = dict(
            steering=float(positive_gain(
                model.readout.steering_log_gain, target_gain_mode).detach()),
            pedals=positive_gain(model.readout.pedal_log_gain, target_gain_mode).detach().tolist())
        anchor_config = config.get('policy_anchor')
        if anchor_config:
            semantic_defaults = dict(pedal_mode='brake_attenuated',
                organ_validity_mode='learned_drive', photometric_normalization=False,
                readout_gain_parameterization='softplus', visual_age_scale=None)
            if (normalization or reconditioned or pedal_conditioning
                    or anchor_config['checkpoint_sha256'] != config['source_checkpoint_sha256']
                    or any(constructor.get(key, default)
                           != source['constructor'].get(key, default)
                           for key, default in semantic_defaults.items())):
                raise ValueError('Policy anchors require unchanged source transduction and codec')
            anchor_coefficient = anchor_config['coefficient']
            anchored_selection_score(dict(conditioned, policy_anchor_mse=0.), anchor_coefficient)
            if anchor_coefficient <= 0:
                raise ValueError('Positive policy preservation coefficient required')
            anchors = prepare_policy_anchors(
                model, episodes, anchor_config['episode_sha256'], chunk, ticks)
            np.savez_compressed(run/'policy-anchor-targets.npz',
                **{key: value.cpu().numpy() for key, value in anchors.items()})
            conditioned['policy_anchor_mse'] = replay_anchor_mse(
                model, episodes, anchors, chunk, ticks)
            write_json(run/'policy-anchor.json', dict(configuration=anchor_config,
                targets_sha256=sha256(run/'policy-anchor-targets.npz'),
                decisions=sum(len(value) for value in anchors.values()),
                interpretation='frozen policy outputs on declared training success episodes; '
                               'not expert labels or held-out evaluation',
                source_initial_anchor_mse=conditioned['policy_anchor_mse']))
        age_confidence = config.get('visual_age_confidence')
        if age_confidence is not None:
            if not anchors or 'visual' not in trainable_groups:
                raise ValueError('Age confidence experiments require frozen source anchors '
                                 'and trainable visual inputs')
            # The original source targets above intentionally precede frontend changes.
            age_report = calibrate_visual_age_decay(
                model, episodes, age_confidence['scale_seconds'])
            constructor['visual_age_scale'] = age_report['scale_seconds']
            conditioned = metrics_for_current_model()
            write_json(run/'visual-age-calibration.json', age_report)
        optimizer = torch.optim.Adam(optimizer_parameters, lr=config["learning_rate"])
        write_json(run/'training-device.json', dict(
            requested_device=str(device), process_id=os.getpid(),
            chunk_validated_training=model.chunk_validated_training,
            device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            parameter_devices=sorted({str(p.device) for p in parameters}),
            observation_devices=sorted({str(v.device) for e in episodes
                                        for v in e['observation'].values()}),
            target_devices=sorted({str(e['action'].device) for e in episodes}),
            training_sparse_backend=constructor.get('sparse_backend', 'coo'),
            deployment_sparse_backend=deployment_backend, torch_version=torch.__version__,
            hip_version=torch.version.hip, precision='float32; no autocast'))

        def apply_gradients():
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite declared-group optimization gradient')
            optimizer.step()

        history, updates = [], 0
        best_metrics, best_epoch = conditioned, 0
        best_state = copy.deepcopy(model.state_dict())
        model.train()
        for epoch in range(config["epochs"]):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started, loss_sum = time.perf_counter(), 0.0
            if update_scope == 'epoch':
                optimizer.zero_grad()
            for batch_index, batch in enumerate(batches):
                state = model.initial_state(len(batch))
                normalizers = torch.tensor(
                    [1 / (len(episodes) * float(control_label_weights(
                        e['action'], e['action_valid'], brake_weight, brake_threshold,
                        coast_weight, coast_threshold).sum()))
                     for e in batch], device=device
                )[None, :, None]
                for offset in range(0, max(len(e["action"]) for e in batch), chunk):
                    obs, labels, valid, active = episode_chunk(batch, offset, chunk)
                    if update_scope == 'chunk':
                        optimizer.zero_grad()
                    prediction, updated = replay_chunk(model, obs, active, state, ticks)
                    weights = control_label_weights(labels, valid, brake_weight, brake_threshold,
                                                    coast_weight, coast_threshold)
                    loss = ((prediction - labels).square() * weights * normalizers).sum()
                    if anchors:
                        loss = loss + anchor_coefficient*policy_anchor_loss(
                            prediction, batch, offset, anchors)
                    loss.backward()
                    if update_scope == 'chunk':
                        apply_gradients()
                        updates += 1
                    # Numerical state survives each optimizer update and gradient detach.
                    state = updated.detach()
                    loss_sum += float(loss.detach())
                    write_json(run / "heartbeat.json", {"epoch": epoch + 1, "batch": batch_index,
                        "offset": offset, "optimizer_updates": updates,
                        "prediction_device": str(prediction.device),
                        "gradient_devices": sorted({str(p.grad.device) for p in parameters
                                                    if p.grad is not None})})
            if update_scope == 'epoch':
                apply_gradients()
                updates += 1
            metrics = metrics_for_current_model()
            # Select by training replay only; never inspect held-out driving for selection.
            if selection_score(metrics) < selection_score(best_metrics):
                best_metrics, best_epoch = metrics, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            record = {
                "epoch": epoch + 1,
                "online_chunk_loss": loss_sum,
                "full_replay": metrics,
                "seconds": time.perf_counter() - started,
            }
            history.append(record)
            write_json(run / "history.json", history)
            print(
                f"epoch={epoch + 1} rmse={metrics['rmse']} seconds={record['seconds']:.2f}",
                flush=True,
            )
        payload = copy.deepcopy(source)
        training_constructor = copy.deepcopy(constructor)
        deployment_constructor = copy.deepcopy(constructor)
        if device.type == 'cuda' or 'deployment_sparse_backend' in config:
            deployment_constructor['sparse_backend'] = deployment_backend
        best_state = {name: value.detach().cpu() for name, value in best_state.items()}
        payload.update(
            model=best_state,
            constructor=deployment_constructor,
            training_constructor=training_constructor,
            training_device=str(device),
            action_profile=manifest["action_profile"],
            training_configuration=config,
            optimizer_updates=updates,
            checkpoint_optimizer_updates=best_epoch * (1 if update_scope == 'epoch' else sum(
                (max(len(e["action"]) for e in batch) + chunk - 1) // chunk for batch in batches)),
            optimizer_update_scope=update_scope,
            selected_epoch=best_epoch,
            selection="minimum mean supervised-action training MSE; optional declared brake "
                      "label weighting and successful-policy preservation; not held-out driving",
            trainable_groups=trainable_groups,
            trainable_parameters=trainable_parameters,
            training_teacher_manifest_sha256=sha256(Path(args.teacher) / "dataset.json"),
            teacher_label_contract=manifest.get('teacher_label_contract'),
            driving_trained="MuJoCo steering proxy only"
            if manifest["action_profile"].startswith("mujoco")
            else "offline labels only; closed-loop performance unverified",
            AC_ready=False,
        )
        torch.save(payload, run / "checkpoint.pt")
        result = {
            "status": "completed",
            "before": before,
            "original_before_photometric_calibration": original,
            "conditioned": conditioned,
            "after": best_metrics,
            "last_epoch": history[-1]["full_replay"],
            "selected_epoch": best_epoch,
            "selection_score_before": selection_score(conditioned),
            "selection_score_after": selection_score(best_metrics),
            "trainable_groups": trainable_groups,
            "training_selection_score_improved": selection_score(best_metrics)
            < selection_score(conditioned),
            "optimizer_updates": updates,
            "optimizer_update_scope": update_scope,
            "policy_anchor": config.get('policy_anchor'),
            "training_device": str(device),
            "chunk_validated_training": model.chunk_validated_training,
            "training_device_name": torch.cuda.get_device_name(device)
                if device.type == 'cuda' else 'CPU',
            "training_sparse_backend": constructor.get('sparse_backend', 'coo'),
            "deployment_sparse_backend": deployment_constructor.get('sparse_backend', 'coo'),
            "optimizer_parameter_devices": sorted({str(p.device) for p in parameters}),
            "gradient_devices": sorted({str(p.grad.device) for p in parameters
                                        if p.grad is not None}),
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == 'cuda' else 0,
            "action_profile": manifest["action_profile"],
            "checkpoint_sha256": sha256(run / "checkpoint.pt"),
            "readout_gain_parameterization": target_gain_mode,
            "gain_before_training": gain_before_training,
            "selected_gain_after_training": dict(
                steering=float(positive_gain(best_state['readout.steering_log_gain'],
                                             target_gain_mode)),
                pedals=positive_gain(best_state['readout.pedal_log_gain'],
                                     target_gain_mode).tolist()),
            "training_rmse_improved_from_conditioned": (
                best_metrics["rmse"][0] < conditioned["rmse"][0]
                if best_metrics["rmse"][0] is not None else None),
            "training_rmse_improved_from_untrained": (
                best_metrics["rmse"][0] < before["rmse"][0]
                if best_metrics["rmse"][0] is not None else None),
            "closed_loop_evaluated": False,
            "AC_ready": False,
            "goal_achieved": False,
        }
        write_json(run / "result.json", result)
    except Exception as error:
        write_json(
            run / "result.json",
            {
                "status": "failed",
                "error": type(error).__name__,
                "message": str(error),
                "goal_achieved": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
