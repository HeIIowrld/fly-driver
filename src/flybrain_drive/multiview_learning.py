"""Offline recurrent distillation of source-specific sensory episodes, never actor route truth."""

import json
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from .episode_storage import load_array, resolve_episode_path
from .graph import sha256
from .multiview_sensory import MultiViewPacker
from .vehicle_circuit import encode_positive_gain, positive_gain

PROFILES = {"normalized_ac_absolute", "mujoco_curvature_proxy_only"}


def configure_trainable_groups(model, groups, organ_groups=None):
    """Explicit adapters/readout only; measured recurrent connections remain fixed."""
    allowed = {'visual', 'organs', 'readout'}
    if (not isinstance(groups, (tuple, list)) or not groups
            or any(not isinstance(g, str) for g in groups)
            or len(set(groups)) != len(groups) or not set(groups) <= allowed):
        raise ValueError('Explicit unique visual/organs/readout training groups required')
    modules = dict(visual=model.adapter.visual_adapter,
                   organs=model.adapter.organ_adapters, readout=model.readout)
    if organ_groups is not None and (
            'organs' not in groups or not isinstance(organ_groups, list) or not organ_groups
            or len(set(organ_groups)) != len(organ_groups)
            or not set(organ_groups) <= set(model.adapter.group_names)):
        raise ValueError('Explicit unique known organ subset requires organs training')
    model.requires_grad_(False)
    for group in groups:
        modules[group].requires_grad_(True)
    if organ_groups is not None:
        for name, adapter in zip(model.adapter.group_names, model.adapter.organ_adapters,
                                 strict=True):
            adapter.requires_grad_(name in organ_groups)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('Selected training groups have no parameters')
    return parameters


def supervised_selection_score(metrics):
    """Mean per-action MSE over supervised heads; missing heads cannot win selection."""
    errors = metrics.get('control_weighted_rmse',
                         metrics.get('brake_weighted_rmse', metrics['rmse']))
    counts = metrics['supervision_counts']
    if len(errors) != 3 or len(counts) != 3:
        raise ValueError('Three action errors and supervision counts required')
    selected = []
    for error, count in zip(errors, counts, strict=True):
        if type(count) is not int or count < 0:
            raise ValueError('Nonnegative integer supervision counts required')
        if count == 0:
            if error is not None:
                raise ValueError('Unsupervised action error must be absent')
        elif error is None or not np.isfinite(error) or error < 0:
            raise ValueError('Finite supervised action errors required')
        else:
            selected.append(error ** 2)
    if not selected:
        raise ValueError('No supervised actions')
    return float(np.mean(selected))


def brake_label_weights(labels, valid, positive_weight=1., threshold=.05):
    """Training-label weighting only; preserve masks, sequence order and actor inputs."""
    if (not np.isfinite([positive_weight, threshold]).all()
            or not 1 <= positive_weight <= 100 or not 0 <= threshold < 1):
        raise ValueError('Finite brake weight 1..100 and label threshold 0..1 required')
    weights = valid.to(dtype=labels.dtype).clone()
    weights[..., 2] *= torch.where(labels[..., 2] > threshold, positive_weight, 1.)
    return weights


def control_label_weights(labels, valid, brake_weight=1., brake_threshold=.05,
                          coast_weight=1., coast_threshold=.005):
    """Offline loss weights only; no actor rule, target change or added observation."""
    if (not np.isfinite([coast_weight, coast_threshold]).all()
            or not 1 <= coast_weight <= 1000 or not 0 <= coast_threshold < 1):
        raise ValueError('Finite coast loss weight1..1000 and threshold0..1 required')
    weights = brake_label_weights(labels, valid, brake_weight, brake_threshold)
    weights[..., 1] *= torch.where(labels[..., 1] <= coast_threshold, coast_weight, 1.)
    return weights


def condition_independent_pedals(readout, output_activity, *, target_rms, max_gain):
    """Fit output scale to sensory replay activity only, never teacher actions."""
    activity = torch.as_tensor(output_activity).detach().double()
    targets = np.asarray(target_rms, dtype=float)
    if (readout.pedal_mode != 'independent' or activity.ndim != 2 or not len(activity)
            or activity.shape[1] != readout.output_size or not torch.isfinite(activity).all()
            or torch.any((activity < 0) | (activity > 1))
            or targets.shape != (2,) or not np.isfinite(targets).all()
            or np.any((targets <= 0) | (targets >= 1))
            or not np.isfinite(max_gain) or not 0 < max_gain <= 1e10):
        raise ValueError('Independent pedals, bounded activity and explicit gain budget required')
    forward = (activity[:, readout.forward_ports]
               @ readout.forward_weights.detach().double().softmax(0))
    halt = activity[:, readout.halt_ports] @ readout.halt_weights.detach().double().softmax(0)
    planned, achieved = [], []
    for drive, target in zip((forward, halt), targets, strict=True):
        def rms(gain, drive=drive):
            return float((-torch.expm1(-gain * drive)).square().mean().sqrt())
        if rms(max_gain) < target:
            raise ValueError('Sensory output activity cannot reach target within gain budget')
        low, high = 0., float(max_gain)
        for _ in range(64):
            middle = (low + high) / 2
            if rms(middle) < target:
                low = middle
            else:
                high = middle
        gain = (low + high) / 2
        planned.append(gain)
        achieved.append(rms(gain))
    mode = getattr(readout, 'gain_parameterization', 'softplus')
    before = positive_gain(readout.pedal_log_gain, mode).detach().tolist()
    raw = encode_positive_gain(readout.pedal_log_gain.new_tensor(planned), mode)
    with torch.no_grad():
        readout.pedal_log_gain.copy_(raw)
    return dict(source='declared training sensory output activity only', teacher_labels_used=False,
        target_rms=targets.tolist(), achieved_rms=achieved, gain_before=before,
        gain_after=planned, max_gain=max_gain,
        interpretation='engineering output scale; not physiological tuning or pedal learning')


def condition_native_steering(readout, output_activity, *, target_rms, max_gain):
    """Label-free native input scale: preserve output IDs/signs and recurrent structure."""
    activity = torch.as_tensor(output_activity).detach().double()
    if (activity.ndim != 2 or not len(activity) or activity.shape[1] != readout.output_size
            or not torch.isfinite(activity).all() or torch.any((activity < 0) | (activity > 1))
            or not np.isfinite([target_rms, max_gain]).all()
            or not 0 < target_rms < 1 or not 0 < max_gain <= 1e10):
        raise ValueError('Finite native steering activity and gain budget required')
    drive = activity[:, readout.steering_ports] @ readout.steering_signs.detach().double()

    def rms(gain):
        return float(torch.tanh(gain*drive).square().mean().sqrt())

    if rms(max_gain) < target_rms:
        raise ValueError('Steering output activity cannot reach target within gain budget')
    low, high = 0., float(max_gain)
    for _ in range(64):
        middle = (low+high)/2
        if rms(middle) < target_rms:
            low = middle
        else:
            high = middle
    gain = (low+high)/2
    mode = getattr(readout, 'gain_parameterization', 'softplus')
    before = float(positive_gain(readout.steering_log_gain, mode).detach())
    with torch.no_grad():
        readout.steering_log_gain.copy_(encode_positive_gain(
            readout.steering_log_gain.new_tensor(gain), mode))
    return dict(source='training sensory output activity only', teacher_labels_used=False,
        gain_before=before, gain_after=gain, target_rms=target_rms, achieved_rms=rms(gain),
        max_gain=max_gain, output_ids_and_signs_changed=False,
        interpretation='engineering scale calibration; physiology and steering learning unverified')


def replay_output_activity(model, episodes, chunk_length, brain_ticks, episodes_per_batch=None):
    """Only graph output neurons, final tick per active decision; numerical state persists."""
    collected, current = [], []
    hook = model.readout.register_forward_pre_hook(
        lambda module, args: current.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            for batch in episode_batches(episodes, episodes_per_batch):
                state = model.initial_state(len(batch))
                for start in range(0, max(len(e['action']) for e in batch), chunk_length):
                    current.clear()
                    obs, _, _, active = episode_chunk(batch, start, chunk_length)
                    _, state = replay_chunk(model, obs, active, state, brain_ticks)
                    values = torch.stack(current)[brain_ticks - 1::brain_ticks]
                    collected.append(values[active.bool()])
    finally:
        hook.remove()
    return torch.cat(collected)


def mixed_legacy_control(teacher, student, beta, rng):
    """Select a DAgger executor after inference; teacher labels never enter the student."""
    teacher = np.asarray(teacher, dtype=np.float32)
    if (teacher.shape != (2,) or not np.isfinite(teacher).all()
            or np.any(np.abs(teacher) > 1) or not np.isfinite(beta) or not 0 <= beta <= 1):
        raise ValueError("Bounded legacy teacher control and mixture probability required")
    if student is None:
        if beta != 1:
            raise ValueError("Student execution requires a student prediction")
        return teacher.copy(), True
    student = np.asarray(student)
    if student.shape != (3,) or not np.isfinite(student).all() or abs(student[0]) > 1:
        raise ValueError("Explicit three-head prediction with bounded steering proxy required")
    use_teacher = bool(rng.random() < beta)
    command = teacher.copy()
    if not use_teacher:
        # Keep only the declared fixed-speed task constraint, not the oracle steering.
        command[1] = student[0]
    return command, use_teacher


def calibrate_photometry(model, episodes, *, rgb_floor, contrast_floor):
    """Label-free moments of each ray in its own source; never pool cameras or episodes' truth."""
    if (not model.photometric_normalization or not episodes
            or not np.isfinite([rgb_floor, contrast_floor]).all()
            or min(rgb_floor, contrast_floor) <= 0):
        raise ValueError("Normalized model, training episodes and positive scale floors required")
    shape = model.photometric_mean.shape
    device = model.photometric_mean.device
    total, squared, count = (torch.zeros(shape, dtype=torch.float64, device=device)
                             for _ in range(3))
    for episode in episodes:
        observation = episode["observation"]
        values = torch.cat([observation["view_rgb"], observation["view_contrast"]], -1).double()
        mask = torch.stack([observation["view_valid"], observation["contrast_valid"]], -1)
        mask = mask.repeat_interleave(3, -1).double()
        total += (values * mask).sum(0)
        squared += (values.square() * mask).sum(0)
        count += mask.sum(0)
    mean = total / count.clamp_min(1)
    floors = torch.tensor([rgb_floor] * 3 + [contrast_floor] * 3, device=device)
    scale = (squared / count.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
    scale = torch.maximum(scale, floors)
    with torch.no_grad():
        model.photometric_mean.copy_(mean)
        model.photometric_scale.copy_(scale)
        model.photometric_seen.copy_((count[..., [0, 3]] > 0).float())
    return {"source": "training observations only", "teacher_labels_used": False,
            "rgb_scale_floor": rgb_floor, "contrast_scale_floor": contrast_floor,
            "valid_rgb_samples_per_view": count[..., 0].sum(1).long().tolist(),
            "constant_validity_channels_used_as_drive": False,
            "unseen_ray_policy": "zero drive", "standardized_clip": 5}


def load_episodes(path, calibration):
    root = Path(path).resolve()
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf8"))
    if (
        manifest.get("schema") != "flybrain.multiview.teacher.v1"
        or manifest.get("action_profile") not in PROFILES
        or manifest.get("sensory_calibration") != calibration
        or set(manifest.get("actor_fields", [])) != MultiViewPacker.fields
    ):
        raise ValueError(
            "Explicit multi-view teacher schema, action profile and matching calibration required"
        )
    episodes, identities = [], set()
    source_coverage = np.zeros(len(calibration["view_order"]), dtype=np.int64)
    for record in manifest["episodes"]:
        file = resolve_episode_path(root, record, manifest)
        if (
            sha256(file) != record["sha256"]
            or record["episode_id"] in identities
        ):
            raise ValueError("Episode path/hash/identity differs from sealed manifest")
        identities.add(record["episode_id"])
        with load_array(file, allow_pickle=False) as data:
            obs = {
                name[4:]: torch.tensor(data[name]) for name in data.files if name.startswith("obs_")
            }
            action, mask, times = data["action"], data["action_valid"], data["observation_time"]
            n = len(action)
            if (
                set(obs) != MultiViewPacker.fields
                or n != record["steps"]
                or not n
                or action.shape != (n, 3)
                or mask.shape != action.shape
                or times.shape != (n,)
                or not np.isfinite(action).all()
                or not np.isfinite(times).all()
                or not np.isin(mask, [0, 1]).all()
                or not mask.any()
                or np.any(np.diff(times) <= 0)
                or np.any(np.abs(action[:, 0]) > 1)
                or np.any((action[:, 1:] < 0) | (action[:, 1:] > 1))
                or any(len(v) != n or not torch.isfinite(v).all() for v in obs.values())
            ):
                raise ValueError("Invalid sensory episode, action supervision or temporal order")
            if manifest["action_profile"] == "mujoco_curvature_proxy_only" and mask[:, 1:].any():
                raise ValueError("MuJoCo curvature proxy cannot supervise native throttle/brake")
            if (obs["view_valid"].ndim != 3
                    or obs["view_valid"].shape[1] != len(source_coverage)
                    or any(not torch.isin(obs[k], obs[k].new_tensor([0, 1])).all()
                           for k in ("view_valid", "contrast_valid", "organ_valid"))):
                raise ValueError("Declared view count and binary sensory validity required")
            source_coverage += obs["view_valid"].bool().any(2).sum(0).numpy()
            episodes.append(
                {
                    "observation": obs,
                    "action": torch.tensor(action, dtype=torch.float32),
                    "action_valid": torch.tensor(mask, dtype=torch.bool),
                    "source": record,
                }
            )
    required = calibration.get('required_views', calibration['view_order'])
    if (not isinstance(required, list) or not required or len(set(required)) != len(required)
            or not set(required) <= set(calibration['view_order'])):
        raise ValueError('Explicit nonempty required view subset required')
    required_indices = [calibration['view_order'].index(name) for name in required]
    if (
        not episodes
        or sum(len(e["action"]) for e in episodes) != manifest["decisions"]
        or np.any(source_coverage[required_indices] == 0)
    ):
        raise ValueError(
            "Nonempty exact decision count and coverage of every declared source required"
        )
    return episodes, manifest


def episode_chunk(episodes, start, length):
    if not episodes or type(start) is not int or start < 0 or type(length) is not int or length < 1:
        raise ValueError("Episodes, nonnegative offset and positive chunk length required")
    prototype = episodes[0]["observation"]
    batch = len(episodes)
    obs = {k: v.new_zeros((length, batch, *v.shape[1:])) for k, v in prototype.items()}
    labels = episodes[0]['action'].new_zeros((length, batch, 3))
    supervision = torch.zeros_like(labels)
    active = labels.new_zeros((length, batch))
    for column, episode in enumerate(episodes):
        n = min(length, max(0, len(episode["action"]) - start))
        if n:
            for key in obs:
                obs[key][:n, column] = episode["observation"][key][start : start + n]
            labels[:n, column] = episode["action"][start : start + n]
            supervision[:n, column] = episode["action_valid"][start : start + n]
            active[:n, column] = 1
    return obs, labels, supervision, active


def replay_chunk(model, observation, active, state, brain_ticks):
    if (
        type(brain_ticks) is not int
        or not 1 <= brain_ticks <= 32
        or active.ndim != 2
        or not torch.isin(active, active.new_tensor([0, 1])).all()
        or any(v.shape[:2] != active.shape for v in observation.values())
        or state.shape != (active.shape[1], model.circuit.size)
    ):
        raise ValueError("Declared brain ticks, activity and persistent state dimensions required")
    length, batch = active.shape
    drive = model.prepare_drive({k: v.flatten(0, 1) for k, v in observation.items()})
    drive = drive.reshape(length, batch, -1)
    chunk_checks = getattr(model, 'chunk_validated_training', False)
    advance = model.advance
    if chunk_checks:
        model.validate_training_chunk(drive, state)
        advance = model.advance_training_chunk
    predictions = []
    for tick in range(length):
        updated = state
        for _ in range(brain_ticks):
            prediction, updated = advance(drive[tick], updated)
        state = torch.where(active[tick, :, None].bool(), updated, state)
        predictions.append(prediction * active[tick, :, None])
    predictions = torch.stack(predictions)
    if chunk_checks:
        model.validate_training_result(predictions, state)
    return predictions, state


def episode_batches(episodes, maximum):
    """Group whole equal-length episodes; no new resets or artificial sequence splitting."""
    if maximum is None:
        return [episodes]
    if type(maximum) is not int or maximum < 1 or not episodes:
        raise ValueError("Positive episode batch maximum and nonempty episodes required")
    result = []
    for _, group in groupby(sorted(episodes, key=lambda e: len(e["action"])),
                            key=lambda e: len(e["action"])):
        bucket = list(group)
        result.extend(bucket[i:i+maximum] for i in range(0, len(bucket), maximum))
    return result


def replay_metrics(model, episodes, chunk_length, brain_ticks, episodes_per_batch=None,
                   brake_positive_weight=1., brake_label_threshold=.05, *, chunk_metric=None,
                   throttle_coast_weight=1., throttle_coast_threshold=.005):
    squared, energy, counts = np.zeros(3), np.zeros(3), np.zeros(3)
    weighted_squared, weighted_counts = np.zeros(3), np.zeros(3)
    decisions, extra_error = 0, 0.
    with torch.no_grad():
        for batch in episode_batches(episodes, episodes_per_batch):
            state = model.initial_state(len(batch))
            for start in range(0, max(len(e["action"]) for e in batch), chunk_length):
                obs, labels, valid, active = episode_chunk(batch, start, chunk_length)
                prediction, state = replay_chunk(model, obs, active, state, brain_ticks)
                if chunk_metric is not None:
                    extra_error = extra_error + chunk_metric(prediction, batch, start)
                squared += ((prediction - labels).square() * valid).sum((0, 1)).cpu().numpy()
                weights = control_label_weights(labels, valid, brake_positive_weight,
                    brake_label_threshold, throttle_coast_weight, throttle_coast_threshold)
                weighted_squared += ((prediction-labels).square()*weights).sum((0, 1)).cpu().numpy()
                weighted_counts += weights.sum((0, 1)).cpu().numpy()
                energy += (prediction.square() * active[..., None]).sum((0, 1)).cpu().numpy()
                counts += valid.sum((0, 1)).cpu().numpy()
                decisions += int(active.sum())
    result = {
        "rmse": [
            float(np.sqrt(v / n)) if n else None for v, n in zip(squared, counts, strict=True)
        ],
        "output_rms": np.sqrt(energy / decisions).tolist(),
        "supervision_counts": counts.astype(int).tolist(),
        "decisions": decisions,
    }
    if brake_positive_weight != 1 or throttle_coast_weight != 1:
        result.update(brake_weighted_rmse=[float(np.sqrt(v/n)) if n else None
            for v, n in zip(weighted_squared, weighted_counts, strict=True)],
            brake_positive_weight=brake_positive_weight,
            brake_label_threshold=brake_label_threshold,
            weighted_supervision_counts=weighted_counts.tolist())
        if throttle_coast_weight != 1:
            result['control_weighted_rmse'] = result.pop('brake_weighted_rmse')
            result.update(throttle_coast_weight=throttle_coast_weight,
                          throttle_coast_threshold=throttle_coast_threshold)
    if chunk_metric is not None:
        extra_value = float(extra_error)
        if not np.isfinite(extra_value) or extra_value < 0:
            raise ValueError('Finite nonnegative extra replay metric required')
        result['extra_chunk_metric'] = extra_value
    return result
